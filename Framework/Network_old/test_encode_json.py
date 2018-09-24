from util import types
from util import encode_json
from typing import Callable
import tensorflow as tf

#defaults = { "activation" : None, "name" : None, "batchsize" : None}
#types = { "activation" : Callable[[tf.Tensor, str], tf.Tensor], "name" : str, 
#        "batchsize" : int }
#requirements = ["activation", "batchsize"]

#defaults = { "d" : None, "batchsize" : None, "stddev_w" : 0.25,
#        "stddev_b" : 0.25, "weight_initializer" : tf.random_normal_initializer,
#        "bias_initializer" : tf.random_normal_initializer }
#types = { "d" : int, "batchsize" : int, "stddev_w" : float, "stddev_b" : float,
#        "weight_initializer" : tf.keras.initializers.Initializer,
#        "bias_initializer" : tf.keras.initializers.Initializer}
#requirements = [ "d", "batchsize"]

defaults = {
        "batchsize" : 1,
        "inputs_dims" : None,
        "train_steps" : None,
        "validate_steps" : None,
        "save_ckpt" : None,
        "save_dir" : None,
        "debug" : True
        }

types = {
        "batchsize" : int,
        "inputs_dims" : tuple,
        "train_steps" : int,
        "validate_steps" : int,
        "save_ckpt" : bool,
        "save_dir" : str,
        "debug" : bool
        }

requirements = [ "inputs_dims", "train_steps", "layers"]

conf = { "defaults" : defaults, "types" : types, "requirements" : requirements }

print(conf)
print(encode_json(conf))

with open("conf/NetworkBuilder.json", "w+") as f:
    f.write(encode_json(conf))
