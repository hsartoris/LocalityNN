from typing import Dict, List
from ..tf_names import *
import tensorflow as tf

param_defaults: Dict[str, any] = {
        'input_shape': None,
        'batchsize': None,
        'activation': relu,
        'dtype': tf.float32,
        'stddev_w': .25,
        'initializer_w': random_normal,
        'stddev_b': .25,
        'initializer_b': random_normal,
        'stddev_w0': None,
        'stddev_w1': None,
        'stddev_w2': None,
        'initializer_w0': None,
        'initializer_w1': None,
        'initializer_w2': None,
        }

param_types: Dict[str, type] = {
        'input_shape': tuple,
        'batchsize': int,
        # TODO: type
        'activation': type(relu),
        # TODO: type
        'dtype': type(tf.float32),
        'stddev_w': float,
        # TODO: type
        'initializer_w': type,
        'stddev_b': float,
        # TODO: type
        'initializer_b': type,
        'stddev_w0': float,
        'stddev_w1': float,
        'stddev_w2': float,
        # TODO: type
        'initializer_w0': type,
        # TODO: type
        'initializer_w1': type,
        # TODO: type
        'initializer_w2': type,
        }

requirements: List[str] = [
        'input_shape',
        'batchsize',
        'stddev_w',
        'stddev_b',
        'initializer_w',
        'initializer_b',
        'dtype',
        'activation'
        ]
