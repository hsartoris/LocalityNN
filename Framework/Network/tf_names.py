import tensorflow as tf
from typing import Callable

#TODO: TYPESSSSSS


# initializers
random_normal: type = tf.random_normal_initializer

# activations
relu: Callable = tf.nn.relu

# kinda hacky but I don't care
names = {
        tf.random_normal_initializer: "random_normal",
        tf.nn.relu: "relu",
}
