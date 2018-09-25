from typing import Dict, List
from ..tf_names import *
import tensorflow as tf

param_defaults: Dict[str, any] = {
        'd': None,
        'input_shape': None,
        'batchsize': 1,
        'stddev_w': .25,
        'stddev_b': .25,
        'weight_initializer': random_normal,
        'bias_initializer': random_normal,
        'activation': relu,
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
        'activation': type(relu),
        'dtype': type(tf.int32)
        }

requirements: List[str] = [
        'd',
        'input_shape'
        ]
