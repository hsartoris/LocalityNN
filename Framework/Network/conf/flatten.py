from typing import Dict, List
from ..tf_names import *
import tensorflow as tf

param_defaults: Dict[str, any] = {
        'input_shape': None,
        'stddev_w': 1.0,
        'initializer_w': random_normal,
        'activation': None,
        'dtype': tf.float32
        }

param_types: Dict[str, type] = {
        'input_shape': tuple,
        'stddev_w': float,
        # TODO: type
        'initializer_w': type,
        # TODO: type
        'activation': type(relu),
        # TODO: type
        'dtype': type(tf.float32)
        }

requirements: List[str] = {
        'input_shape',
        'stddev_w',
        'initializer_w',
        'dtype'
        }
