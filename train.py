import json
import os
import logging
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import sys
import datetime

#sys.path.append('/')

import numpy as np
import argparse

import tensorflow as tf
tf.get_logger().setLevel('ERROR')

from tensorflow.keras.mixed_precision import experimental as mixed_precision

from presearch_trrosetta.architecture.trRosetta import trRosetta
from presearch_trrosetta.utils import optimization
from presearch_trrosetta.utils.config import DistanceConfig
from presearch_trrosetta.utils.dataset import load_dataset

# use only one gpu.
# if you want to use multiple gpu, please delete two line.
physical_devices = tf.config.list_physical_devices('GPU')
tf.config.set_visible_devices(physical_devices[0], 'GPU')

#tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

# https://github.com/tensorflow/models/blob/9e46a6cbf30ba0d80dd7a185e1e93899bea7e6d7/official/nlp/transformer/metrics.py#L166
class MetricLayer(tf.keras.layers.Layer):
    def __init__(self, max_len, name='metric_layer'):
        self.metric_fns = []

        self.range_mask = np.zeros([3, max_len, max_len, 1])
        for i in range(max_len):
            self.range_mask[0, i, i:i + 12, 0] = 1
            self.range_mask[1, i, i + 12:i + 24, 0] = 1
            self.range_mask[2, i, i + 24:, 0] = 1

        super(MetricLayer, self).__init__(name=name)

    def build(self, input_shape):
        self.precision_short = tf.keras.metrics.Precision(name="precision_short")
        self.recall_short = tf.keras.metrics.Recall(name="recall_short")
        self.precision_medium = tf.keras.metrics.Precision(name="precision_medium")
        self.recall_medium = tf.keras.metrics.Recall(name="recall_medium")
        self.precision_long = tf.keras.metrics.Precision(name="precision_long")
        self.recall_long = tf.keras.metrics.Recall(name="recall_long")

        # |i-j|< 12      ->short term
        # 12< |i -j|<24  ->medium term
        # 24 < |i-j|     ->long term

    def call(self, gt, pred, mask):
        pred_argmax = tf.argmax(pred, axis=-1)
        gt_argmax = tf.argmax(gt, axis=-1)

        pred_contact = pred_argmax <= 6
        gt_contact = gt_argmax <= 6

        self.precision_short(gt_contact, pred_contact, self.range_mask[0] * mask)
        self.recall_short(gt_contact, pred_contact, self.range_mask[0] * mask)

        self.precision_medium(gt_contact, pred_contact, self.range_mask[1] * mask)
        self.recall_medium(gt_contact, pred_contact, self.range_mask[1] * mask)

        self.precision_long(gt_contact, pred_contact, self.range_mask[2] * mask)
        self.recall_long(gt_contact, pred_contact, self.range_mask[2] * mask)

        return pred

# https://github.com/tensorflow/models/blob/2ad3e213838f71e92af198d917ac5574c9d60294/official/nlp/bert/bert_models.py#L173
# Reference - tensorflow code style
def create_train_model(config,
                       is_training=True):
    # config
    max_len = config.max_len
    bins = config.bins

    # input_seq = tf.keras.layers.Input(
    #     shape=([max_len]), name='seq', dtype=tf.int32)
    # input_msa = tf.keras.layers.Input(
    #     shape=([max_len, max_len, 442]), name='f2d_dca', dtype=tf.float64)
    # input_pssm = tf.keras.layers.Input(
    #     shape=([max_len, 21]), name='f1d_pssm', dtype=tf.float64)
    input_mask = tf.keras.layers.Input(
        shape=[max_len, max_len, 1], name='mask', dtype=tf.int32)
    input_target = tf.keras.layers.Input(
        shape=[max_len, max_len, bins], name='target', dtype=tf.int32)

    # config
    # maybe we don't change n2d_filters and dropout_rate.
    kwargs = dict(
        max_len=config.max_len,
        bins=config.bins,
        n2d_filters=64,
        n2d_layers=61,
        dropout_rate=0.15,
    )

    model = trRosetta(**kwargs)

    # todo : when we use model.summary(), it can't be checked the detail about how this model consist of Layer.
    # output = model(inputs=[input_seq,input_msa,input_pssm])

    # metric
    output = MetricLayer(max_len)(input_target, model.output, input_mask)

    # re-create Model
    model = tf.keras.Model(inputs=model.inputs + [input_target] + [input_mask], outputs=output)

    # loss
    loss_object = tf.losses.CategoricalCrossentropy(reduction=tf.keras.losses.Reduction.NONE)
    loss = loss_object(input_target, output, sample_weight=input_mask)
    model.add_loss(loss)


    return model

# reference : https://www.tensorflow.org/official_models/fine_tuning_bert#optimizers_and_schedules
def bert_optimizer(config):
    # schedule params
    learning_rate = config.learning_rate
    total_step = config.epochs * config.step
    warmup_step = config.warmup_step
    weight_decay_rate = config.weight_decay_rate
    end_learning_rate = config.end_learning_rate

    decay_schedule = tf.keras.optimizers.schedules.PolynomialDecay(initial_learning_rate=learning_rate,
                                                                   decay_steps=total_step,
                                                                   end_learning_rate=end_learning_rate)

    warmup_schedule = optimization.WarmUp(initial_learning_rate=learning_rate,
                                          decay_schedule_fn=decay_schedule,
                                          warmup_steps=warmup_step)

    optimizer = optimization.AdamWeightDecay(learning_rate=warmup_schedule,
                                             weight_decay_rate=weight_decay_rate,
                                             epsilon=1e-6,
                                             exclude_from_weight_decay=['LayerNorm', 'layer_norm', 'bias'])
    return optimizer


# https://stackoverflow.com/questions/43784921/how-to-display-custom-images-in-tensorboard-using-keras
class PrintLR(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        print('\n epoch : {}   lr : {}'.format(epoch + 1,self.model.optimizer.lr(self.model.optimizer.iterations)))

def create_callback(config):
    folder = config.experiment_folder
    
    current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    
    # ckpt '{epoch:02d}-{val_loss:.5f}.h5'
    #
    os.makedirs(f'{folder}/logs/ckpt/',exist_ok=True)
    filename = f'{folder}/logs/ckpt/'+'{epoch:02d}-{loss:.5f}-{val_loss:.5f}.h5'
    ckpt_callback = tf.keras.callbacks.ModelCheckpoint(filepath=filename,
                                                       save_weights_only=True)
    #logging.info(f"ckpt file format {filename}")

    # tensorboard
    log_dir = f'{folder}/logs/gradient_tape/' + current_time
    tensorboard_callback = tf.keras.callbacks.TensorBoard(
            log_dir,)
#            histogram_freq = 1,
#            profile_batch = '20, 40')

    # I want to see the value of learning rate while training.
    lr_callback = PrintLR()

    callback =[lr_callback,
               ckpt_callback,
               tensorboard_callback]
    
    
    return callback

def create_folder(config):
    """
    when evaluate, we want to save the result
    """
    folder = config.experiment_folder

    os.makedirs(f'{folder}/rr', exist_ok=True)
    os.makedirs(f'{folder}/gt_pred', exist_ok=True)
    os.makedirs(f'{folder}/pred', exist_ok=True)


def train(config,
          mode='train'):
    """
    for training or evaluation.

    1. create model
    2. create dataset
    3. compile
    4. fit or eval

    """

    # train params 
    epochs = config.epochs
    steps_per_epoch = config.step
    
    with strategy.scope():
        optimizer = bert_optimizer(config)
        model = create_train_model(config)
        model.compile(optimizer=optimizer)

    #model.summary()
    callback = create_callback(config)

    initial_epoch = 0
    if config.load_weight is not None :
        model.load_weights(filepath=config.load_weight)
        logging.info(f" load weight : {config.load_weight}")
        name = os.path.basename(config.load_weight)
        initial_epoch = int(os.path.splitext(name)[0].split('-')[0])
        model.optimizer.iterations.assign(initial_epoch*steps_per_epoch)
        logging.info(f" epoch : {initial_epoch}")

    # train
    if mode == 'train':
        with strategy.scope():
            train_tfrecords = [f'{config.train_path}/{file}' for file in os.listdir(config.train_path) if
                               'tf' in file]
            valid_tfrecords = [f'{config.valid_path}/{file}' for file in os.listdir(config.valid_path) if
                               'tf' in file]

            train_dataset = load_dataset(train_tfrecords, config)
            valid_dataset = load_dataset(valid_tfrecords, config)

        model.fit(train_dataset,
                  validation_steps=200,
                  validation_data=valid_dataset,
                  epochs = epochs,
                  steps_per_epoch = steps_per_epoch,
                  callbacks = callback,
                  initial_epoch= initial_epoch,
                 )
    else :
        with strategy.scope():
            eval_tfrecords = [f'{config.eval_path}/{file}' for file in os.listdir(config.eval_path) if
                               'tf' in file]
            eval_dataset = load_dataset(eval_tfrecords, config, is_training=False)
        # todo : value, gt+pred, pred, rr
        #create_folder(config)
        model.evaluate(eval_dataset)

                

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('-e','--experiment_folder',default=None)
    parser.add_argument('--eval', action='store_true')
    parser.add_argument('--log', action='store_true')
    args = parser.parse_args()

    # set logger
    if args.log:
        logging.basicConfig(level=logging.INFO)

    os.makedirs(args.experiment_folder,exist_ok=True)
    if args.experiment_folder == None :
        config = DistanceConfig()
    else : 
        if os.path.isfile(f'{args.experiment_folder}/config.json'):
            logging.info(" config file exist")
            config = DistanceConfig.from_json_file(f'{args.experiment_folder}/config.json')
        else : 
            config = DistanceConfig()
            with open(f'{args.experiment_folder}/config.json', 'w', encoding='utf-8') as config_file :
                json.dump(config.to_dict(), config_file)




    policy_name = 'mixed_float16'
    policy = mixed_precision.Policy(policy_name)
    mixed_precision.set_policy(policy)

    strategy = tf.distribute.MirroredStrategy()

    config.experiment_folder = args.experiment_folder
    config.load_weight = f'{args.experiment_folder}/{config.load_weight}'

    if not args.eval :
        train(config )
    else :
        train(config, mode='eval')
