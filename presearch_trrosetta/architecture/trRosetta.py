import tensorflow as tf


# todo : check the abstract method, but it is not needed. maybe ? -> call, get_config
class trRosetta(tf.keras.Model) :
    """
    trRosetta network.

    Args :
        max_len : maximum length of fasta
        n2d_layes : how many use residual block?
        n2d_filters : number of CNN channel.
        dropout_rate : dropout. default value is 0.15
        bins : dimension of output. default value is 16. it means that range of distance is 2~18 Ångström.
               "18" means that distance is longer than 18 Ångström.
                This case is very hard to predict, it seems to be impossible prediction.
                because long distance means no interaction.
    """

    def __init__(self,
                 max_len = 300,
                 n2d_layers=61,
                 n2d_filters  = 64,
                 dropout_rate=0.15,
                 bins=16,
                 **kwargs):


        input_seq = tf.keras.layers.Input(shape=([max_len]), dtype=tf.int32, name='seq')
        input_f2d_dca = tf.keras.layers.Input(shape=([max_len, max_len, 442]), name='f2d_dca')
        input_f1d_pssm = tf.keras.layers.Input(shape=([max_len, 21]), name='f1d_pssm')

        f1d = tf.one_hot(input_seq, 21)
        f1d = tf.concat(values=[f1d, input_f1d_pssm], axis=-1)
        f1d = tf.reshape(f1d, [-1, max_len, 42])
        # todo : check
        f2d = tf.concat([tf.tile(f1d[:, :, None, :], [1, 1, max_len, 1]),
                         tf.tile(f1d[:, None, :, :], [1, max_len, 1, 1]),
                         input_f2d_dca], axis=-1)

        f2d = tf.reshape(f2d, [-1, max_len, max_len, 442 + 2 * 42])
        input_x = f2d

        x = tf.keras.layers.Conv2D(n2d_filters, 1)(input_x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.ELU()(x)

        # stack of residual blocks with dilations
        dilation = 1
        for _ in range(n2d_layers):
            last_x = x
            x = tf.keras.layers.Conv2D(n2d_filters, 3, padding='same', dilation_rate=dilation)(x)
            x = tf.keras.layers.BatchNormalization()(x)
            x = tf.keras.layers.ELU()(x)
            x = tf.keras.layers.Dropout(rate=dropout_rate)(x)
            x = tf.keras.layers.Conv2D(n2d_filters, 3, padding='same', dilation_rate=dilation)(x)
            x = tf.keras.layers.BatchNormalization()(x)
            x = tf.keras.layers.ELU()(x + last_x)

            dilation *= 2
            if dilation > 16:
                dilation = 1

        outputs = tf.keras.layers.Conv2D(bins, 1, activation='softmax')(x)

        inputs = {"seq" : input_seq ,
                  "f2d_dca" : input_f2d_dca,
                  "f1d_pssm" : input_f1d_pssm,
        }

        super(trRosetta, self).__init__(
            inputs=inputs, outputs=outputs, **kwargs)