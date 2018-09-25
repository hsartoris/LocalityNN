from typing import Dict, List
import tensorflow as tf

param_defaults: Dict[str, any] = {
        'd': None,
        'input_shape': None,
        'batchsize': 1,
        'stddev_w': .25,
        'stddev_b': .25,
        'weight_initializer': tf.random_normal_initializer,
        'bias_initializer': tf.random_normal_initializer,
        'dtype': tf.float32
        }

param_types: Dict[str, type] = {
        'd': int,
        'input_shape': tuple,
        'batchsize': int,
        'stddev_w': float,
        'stddev_b': float,
        #'weight_initializer': tf.keras.initializers.Initializer,
        #'bias_initializer': tf.keras.initializers.Initializer
        'weight_initializer': type,
        'bias_initializer': type,
        'dtype': type(tf.int32)
        }

requirements: List[str] = [
        'd',
        'input_shape'
        ]
