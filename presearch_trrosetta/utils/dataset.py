import tensorflow as tf
import numpy as np 
import logging


# Reference - tensorflow code style.
# https://github.com/tensorflow/models/blob/2ad3e213838f71e92af198d917ac5574c9d60294/official/nlp/bert/input_pipeline.py#L151
def load_dataset(file,
                 config,
                 is_training =True):
    max_len = config.max_len
    batch_size = config.batch_size
    buffer_size = config.buffer_size
    bins = config.bins

    features = {
        'fasta': tf.io.FixedLenFeature([], tf.string),
        'seq': tf.io.RaggedFeature(value_key="seq", dtype=tf.int64),
        'length': tf.io.FixedLenFeature([1], tf.int64),
        'f2d_dca': tf.io.FixedLenFeature([], tf.string),
        'f1d_pssm': tf.io.FixedLenFeature([], tf.string),
        'target': tf.io.FixedLenFeature([], tf.string),
        'num_a3m': tf.io.FixedLenFeature([1], tf.int64),

    }

    dataset = tf.data.TFRecordDataset(file,
                                      num_parallel_reads=tf.data.experimental.AUTOTUNE, )

    if is_training :
        dataset = dataset.shuffle(buffer_size)
        dataset = dataset.repeat()
    else :
        logging.info("you are using dataset for evaluation. no repeat, no shuffle")

    # only upper triangular matrix (excluded diagonal matrix)
    triangle_mat = np.triu(np.ones([max_len, max_len]), k=1)

    def _parse_function(example_proto):
        eg = tf.io.parse_single_example(example_proto, features)
        length = tf.cast(eg['length'][0], dtype=tf.int64)
        mask = tf.constant(triangle_mat)[:length, :length]
        target = tf.io.parse_tensor(eg['target'], out_type=tf.int32)

        # distance range is 2~18
        # clipping range is 0~16
        target = tf.clip_by_value(target, 2, bins + 1) - 2

        # if you use sparse categorical cross entropy, converting to one hot vector is not needed.
        target = tf.one_hot(target, bins)

        return {
            'fasta': eg['fasta'],
            'seq': eg['seq'],
            'length': length,
            'f2d_dca': tf.io.parse_tensor(eg['f2d_dca'], out_type=tf.float32),
            'f1d_pssm': tf.io.parse_tensor(eg['f1d_pssm'], out_type=tf.float32),
            'target': target,
            'mask': tf.expand_dims(mask, axis=-1),
            'num_a3m' : eg['num_a3m'][0],
        }

    padded_shapes = {
        'fasta': [],
        'seq': [max_len],
        'length': [],
        'f2d_dca': [max_len, max_len, 442],
        'f1d_pssm': [max_len, 21],
        'target': [max_len, max_len, bins],
        'mask': [max_len, max_len, 1],
        'num_a3m':[]
    }

    dataset = dataset.map(
        _parse_function,
        num_parallel_calls=tf.data.experimental.AUTOTUNE)
    dataset = dataset.padded_batch(batch_size, padded_shapes=padded_shapes)
    dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)

    return dataset
