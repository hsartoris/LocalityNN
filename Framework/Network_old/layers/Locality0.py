from ..AbstractLayer import AbstractLayer
from ..util import make_expand
import tensorflow as tf
import numpy as np
from typing import Dict, Callable, List, Tuple

class Locality0(AbstractLayer):
    """Modular implementation of first locality-based layer.

    Expected inputs are matrices of shape (timesteps x num_neurons), from which 
    the layer will derive most of its dimensionality.

    If using a batchsize of 1, the layer will still expect the inputs to have an 
    extra dimesion of size 1.

    Parameters:
        d: int - chosen output depth. Defaults to timesteps/2
        batchsize: int - number of samples per batch
        stddev_w: float - standard deviation of weight matrices
        stddev_b: float - standard deviation of bias matrices
        weight_initializer: tf.keras.initializers.Initializer
        bias_initializer: tf.keras.initializers.Initializer

    """
    # TODO: add explanation of what the hell it actually does.

    def _setup(self) -> None:
        """Uses input dimensions to instantiate weight and bias matrices, as 
        well as required tiling matrices.
        """
        self.tsteps: int = self.input_shape[1]
        assert(not self.tsteps == 0)
        assert(not self.n == 0)

        self.d: int = self.params['d']

        # initialize weights
        self.W: tf.Tensor = tf.get_variable("weights",
                shape=(self.d, 2 * self.tsteps),
                dtype = self.dtype,
                initializer = self.params['weight_initializer'])

        # initialize biases
        self.B: tf.Tensor = tf.get_variable("biases", shape=(1, self.d, 1),
                dtype = self.dtype,
                initializer = self.params['bias_initializer'])

    def _layer_ops(self) -> tf.Tensor:
        """Returns input data with locality operations applied to it.
        """
        # TODO: potentially scope this better

        # expand data by duplicating columns side by side
        top_expand: tf.Tensor = tf.einsum('ijk,kl->ijl', self.inputs, 
            self.expand)
        # tile entire inputs n times
        bot_tile: tf.Tensor = tf.tile(self.inputs, [1, 1, self.n])

        # stack results to form (2*timesteps x n^2) matrix
        stack: tf.Tensor = tf.concat([top_expand, bot_tile], 1)

        # apply convolutional weights
        conv_out: tf.Tensor = tf.einsum('ij,kjl->kil', self.W, stack)

        # add bias, tiled to cover entire matrix
        bias_tile: tf.Tensor = tf.tile(self.B, [self.batchsize, 1,
            self.n * self.n])
        bias_out: tf.Tensor = tf.add(conv_out, bias_tile)

        # return biased values with activation applied
        return self.activation(bias_out)

    #@classmethod
    #def _get_default_params(cls) -> Dict[str,Tuple[any, type]]:
    #    """Returns default parameters for this layer module."""
    #    return {
    #            'd': (None, int), # must be provided
    #            'batchsize': (1, int),
    #            'stddev_w': (.25, float),
    #            'stddev_b': (.25, float),
    #            'weight_initializer': (tf.random_normal_initializer, 
    #                tf.keras.initializers.Initializer),
    #            'bias_initializer': (tf.random_normal_initializer,
    #                tf.keras.initializers.Initializer)
    #            }

    def _get_global_constants(self) -> None:
        """Gets or creates `expand` matrix.
        
        Because multiple `expand` matrices may eventually exist, the global name 
        is expandN, where N is the base number of nodes defining the size.
        """
        # at this point _setup has not been called, so we pull the number of 
        # neurons from self.input_shape
        self.n: int = self.input_shape[2]
        with tf.variable_scope(tf.get_variable_scope(), reuse=tf.AUTO_REUSE):
            self.expand: tf.Tensor = tf.get_variable(
                    "expand" + str(self.input_shape[2]),
                    shape = (self.n, self.n * self.n),
                    dtype = self.dtype,
                    initializer = tf.constant_initializer(make_expand(self.n))
                    )
        
    def _validate_params(self) -> None:
        # has d been provided
        if self.params['d'] is None or type(self.params['d']) is not int:
            raise AttributeError("""First output dimension `d` is not provided 
                    or is not an integer.""")

    def required_params(self) -> List[str]:
        """Returns keys in `params` dict for which values must be provided.
        """
        return ['d']
