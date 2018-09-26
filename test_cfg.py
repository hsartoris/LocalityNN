from Framework.Network.layers import *
from Framework.Network.tf_names import *

network_params = {
    'batchsize': 1,
    'data_shape': (2, 3),
    'init_learn_rate': None,
    'train_steps': None,
    'save_dir': None,
    'stack_params': {
        'batchsize': 1,
        'debug': True,
        'layers': [
            (Locality0, 'Locality0',
                {
                    'd': 6,
                    'input_shape': (1, 2, 3),
                    'batchsize': 1,
                    'stddev_w': 0.25,
                    'stddev_b': 0.25,
                    'weight_initializer': random_normal,
                    'bias_initializer': random_normal,
                    'activation': relu,
                    'dtype': tf.float32,
                }
            ),
            (Locality1, 'Locality1',
                {
                    'input_shape': (1, 6, 9),
                    'batchsize': 1,
                    'activation': relu,
                    'dtype': tf.float32,
                    'stddev_w': 0.25,
                    'initializer_w': random_normal,
                    'stddev_b': 0.25,
                    'initializer_b': random_normal,
                    'stddev_w0': None,
                    'stddev_w1': None,
                    'stddev_w2': None,
                    'initializer_w0': None,
                    'initializer_w1': None,
                    'initializer_w2': None,
                }
            ),
            (Locality1, 'Locality1_0',
                {
                    'input_shape': (1, 6, 9),
                    'batchsize': 1,
                    'activation': relu,
                    'dtype': tf.float32,
                    'stddev_w': 0.25,
                    'initializer_w': random_normal,
                    'stddev_b': 0.25,
                    'initializer_b': random_normal,
                    'stddev_w0': None,
                    'stddev_w1': None,
                    'stddev_w2': None,
                    'initializer_w0': None,
                    'initializer_w1': None,
                    'initializer_w2': None,
                }
            ),
            (Flatten, 'Flatten',
                {
                    'input_shape': (1, 6, 9),
                    'stddev_w': 1.0,
                    'initializer_w': random_normal,
                    'activation': None,
                    'dtype': tf.float32,
                }
            ),
        ]
    }
}
