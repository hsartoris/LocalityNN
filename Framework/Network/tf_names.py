import tensorflow as tf
from typing import Callable
from .custom_ops import modified_mse

#TODO: TYPESSSSSS

# jesus christ what is this
Optimizer: type = type

# losses
absolute_difference: Callable = tf.losses.absolute_difference

# initializers
random_normal: type = tf.random_normal_initializer

# activations
relu: Callable = tf.nn.relu
softmax: Callable = tf.nn.softmax

# optimizers
Adadelta: Optimizer = tf.keras.optimizers.Adadelta
Adagrad: Optimizer = tf.keras.optimizers.Adagrad
Adam: Optimizer = tf.train.AdamOptimizer
Adamax: Optimizer = tf.keras.optimizers.Adamax
Nadam: Optimizer = tf.keras.optimizers.Nadam
RMSprop: Optimizer = tf.keras.optimizers.RMSprop
SGD: Optimizer = tf.keras.optimizers.SGD

# kinda hacky but I don't care
names = {
        tf.random_normal_initializer: "random_normal",
        tf.nn.relu: "relu",
        tf.nn.softmax: "softmax",
        tf.keras.optimizers.Adadelta: "Adadelta",
        tf.train.AdamOptimizer: "Adam",
        tf.keras.optimizers.Adagrad: "Adagrad",
        tf.keras.optimizers.Adamax: "Adamax",
        tf.keras.optimizers.Nadam: "Nadam",
        tf.keras.optimizers.RMSprop: "RMSprop",
        tf.keras.optimizers.SGD: "SGD",
        modified_mse: "modified_mse",
        tf.losses.absolute_difference: "absolute_difference"
}
