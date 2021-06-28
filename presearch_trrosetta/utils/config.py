import copy
import json
import six
import tensorflow as tf

# Reference - tensorflow code style.
# https://github.com/tensorflow/models/blob/8ccc242cba24f1f7628545c11b40e3be9a0e0e31/official/nlp/bert/configs.py
class DistanceConfig : 
    
    def __init__(self,
                 train_path = '/J/data/tfrecord/train',
                 valid_path='/J/data/tfrecord/valid',
                 eval_path='/J/data/tfrecord/eval',
                 max_len = 300,
                 random_seed = 12345,
                 validation_ratio = 0.2,
                 batch_size= 1,
                 epochs = 1000,
                 step = 1000,
                 load_weight= None,
                 learning_rate = 1e-3,
                 end_learning_rate = 1e-7,
                 warmup_step = 10000,
                 bins = 16,
                 weight_decay_rate = 1e-5,
                 buffer_size = 30,
                ):
        
        self.validation_ratio = validation_ratio
        
        self.train_path = train_path
        self.valid_path = valid_path
        self.eval_path = eval_path

        
         # Scheduler 
        self.learning_rate = learning_rate
        self.end_learning_rate = end_learning_rate
        self.warmup_step = warmup_step
        self.weight_decay_rate = weight_decay_rate

        
        # train
        self.max_len = max_len
        self.random_seed = random_seed
        self.batch_size = batch_size
        self.epochs = epochs
        self.step = step 
        self.buffer_size = buffer_size
        
        # Load 
        self.load_weight = load_weight
        
        self.bins = bins

    @classmethod
    def from_dict(cls, json_object):
        config = DistanceConfig()
        for (key, value) in six.iteritems(json_object):
            config.__dict__[key] = value
        return config

    @classmethod
    def from_json_file(cls, json_file):
        with tf.io.gfile.GFile(json_file, "r") as reader:
            text = reader.read()
        return cls.from_dict(json.loads(text))

    def to_dict(self):    
        output = copy.deepcopy(self.__dict__)
        return output

    def to_json_string(self):
        return json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n"
