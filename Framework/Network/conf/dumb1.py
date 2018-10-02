from ..tf_names import *
from typing import Dict, List, Tuple
import tensorflow as tf

param_defaults: Dict[str, any] = {
        'input_shape': None,
        'batchsize': None,
        'activation': relu,
        'dtype': tf.float32,
        'stddev_w': 1.0,
        'initializer_w': random_normal,
        'stddev_b': 1.0,
        'initializer_b': random_normal,
        }

param_types: Dict[str, type] = {
        'input_shape': tuple,
        'batchsize': int,
        'activation': type(relu),
        'dtype': type(tf.float32),
        'stddev_w': float,
        'initializer_w': type,
        'stddev_b': float,
        'initializer_b': type
        }

requirements: List[str] = [
        'input_shape',
        'batchsize'
        ]
