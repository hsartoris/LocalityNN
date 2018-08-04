from ..AbstractLayer import AbstractLayer
import tensorflow as tf
import numpy as np
from typing import Dict, Callable, List

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
        expand: (n x n^2) matrix to expand by duplicating columns

    """
    # TODO: add explanation of what the hell it actually does.

    @classmethod
    def _get_default_params(cls) -> Dict[str,any]:
        """Returns default parameters for this layer module."""
        return {
                'd': None, # must be provided
                'batchsize': 1,
                'stddev_w': .25,
                'stddev_b': .25,
                'weight_initializer': tf.random_normal_initializer,
                'bias_initializer': tf.random_normal_initializer,
                'expand': None # must be provided
                }

    @classmethod
    def _global_consts(cls) -> List[str]:
        """Instructs the managing object to create a constant outside the scope 
        of this object, for reuse by other layers if necessary.
        """
        return ['expand']

    def _validate_params(self) -> None:
        # has `expand` been provided?
        if self.params['expand'] is None:
            raise AttributeError("`expand` has not been provied.")
        # if the layer has calculated n, is it compatible?
        if hasattr(self, 'n'):
            assert(self.params['expand'].get_shape() == (self.n, self.n**2))

        # has d been provided
        if self.params['d'] is None or type(self.params['d']) is not int:
            raise AttributeError("""First output dimension `d` is not provided 
                    or is not an integer.""")



    def _setup(self) -> None:
        """Uses input dimensions to instantiate weight and bias matrices, as 
        well as required tiling matrices.
        """
        self.tsteps: int = self.input_shape[1]
        self.n: int = self.input_shape[2]
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

        self.expand: tf.Tensor = self.params['expand']


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


