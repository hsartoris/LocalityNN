from Framework.Network.layers import *
from Framework.Network.tf_names import *

params = {
    'batchsize': 1,
    'input_dims': (1, 1, 1),
    'debug': True,
    'layers': [
        (Locality0, 'test',
            {
                'd': 3,
                'input_shape': (1, 1, 1),
                'batchsize': 1,
                'stddev_w': 0.25,
                'stddev_b': 0.25,
                'weight_initializer': random_normal,
                'bias_initializer': random_normal,
                'dtype': tf.float32,
            }
        ),
    ]
}
