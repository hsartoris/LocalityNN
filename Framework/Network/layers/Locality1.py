from ..AbstractLayer import AbstractLayer
from ..util import make_expand
import tensorflow as tf
import numpy as np
from typing import Dict, Callable, List, Tuple
from math import sqrt

class Locality1(AbstractLayer):
    """Modular implementation of second locality-based layer; the meat of the 
    operation.

    Expected inputs are matrices of shape (batchsize x d x num_neurons), from 
    which the layer will derive most of its dimensionality.

    Parameter defaults are defined in ../conf/locality1.py

    Parameters:
        input_shape: Tuple[int,int,int] - dimensions of input data
        batchsize: int - number of samples per batch
        stddev_w: float - standard deviation of weights. sets stddev for all 
            weight matrices without specific standard deviations
        stddev_w0: float - standard deviation of weight matrix 0. overrides 
            stddev_w
        stddev_w1: float - standard deviation of weight matrix 1. overrides 
            stddev_w
        stddev_w2: float - standard deviation of weight matrix 2. overrides 
            stddev_w
        initializer_w: Initializer - default initializer for all weight matrices
        initializer_w0: Initializer - overrides initializer for weight matrix 0
        initializer_w1: Initializer - overrides initializer for weight matrix 1
        initializer_w2: Initializer - overrides initializer for weight matrix 2
        stddev_b: float - standard deviation of bias matrix
        initializer_b: Initializer - initializer for bias matrix
        dtype: type - type of input data
        activation: Activation - activation used at end of layer
    """

    def _setup(self) -> None:
        """Uses input dimensions to instantiate weight and bias matrices, as 
        well as expand matrix.
        """
        #TODO: type
        self.activation = self.params['activation']

        self.batchsize: int = self.params['batchsize']

        self.input_shape: Tuple[int,int,int] = self.params['input_shape']

        self.d: int = self.input_shape[1]
        assert(not self.d == 0)

        self.n: int = int(sqrt(self.input_shape[2]))
        assert(not self.n == 0)

        self.dtype: type = self.params['dtype']

        # initialize weights
        self.W: List[tf.Tensor] = []
        
        for i in range(3):
            #TODO: type
            w_init = self.params['initializer_w']
            if self.params['initializer_w' + str(i)] is not None:
                w_init = self.params['initializer_w' + str(i)]

            w_stddev: float = self.params['stddev_w']
            if self.params['stddev_w' + str(i)] is not None:
                w_stddev = self.params['stddev_w' + str(i)]

            self.W.append(tf.get_variable("weights_" + str(i),
                    shape = (self.d, (self.d if i < 2 else 2 * self.d)),
                    dtype = self.dtype,
                    initializer = w_init(stddev=w_stddev)))

        #TODO: type
        b_init = self.params['initializer_b']
        b_stddev = self.params['stddev_b']
        self.B: tf.Tensor = tf.get_variable("biases", shape = (1, self.d, 1),
                dtype = self.dtype,
                initializer = b_init(stddev=b_stddev))

        # create/load expand matrix
        with tf.variable_scope(tf.get_variable_scope(), reuse=tf.AUTO_REUSE):
            self.expand: tf.Tensor = tf.get_variable(
                    "expand" + str(self.n),
                    shape = (self.n, self.n * self.n),
                    dtype = self.dtype,
                    initializer = tf.constant_initializer(make_expand(self.n)))

        # create/load tile matrix
        tile: np.ndarray = np.array([([1] + [0]*(self.n-1))*self.n])
        for i in range(1, self.n):
            tile = np.append(tile, [([0]*i + [1] + [0]*(self.n-1-i))*self.n], 0)

        self.tile: tf.Tensor = tf.constant(tile, self.dtype)

    def _compute_layer_ops(self) -> tf.Tensor:
        # (d x n^2) matrix
        # imagine it as a (d x n x n) rectangular solid
        
        # maps (d x n x n) -> (d x n x 1) by summing and averaing (I think)
        horizCompress: tf.Tensor = tf.einsum('ijk,kl->ijl', self.inputs,
                tf.transpose(self.expand))
        horizCompress = tf.divide(horizCompress, self.n)

        # tile out (d x n x 1) -> (d x n x n)
        horiz: tf.Tensor = tf.einsum('ijk,kl->ijl', horizCompress, self.tile)

        # multiply elementwise with input data to force using information
        # TODO: does this make sense
        in_total: tf.Tensor = horiz * self.inputs
        # apply some processing to this chunk
        in_total = tf.einsum('ij,ljk->lik', self.W[0], in_total)

        # maps (d x n x n) -> (d x 1 x n) by summing and averaging
        # honestly this may be switched with horizCompress but it's unimportant
        vertCompress: tf.Tensor = tf.einsum('ijk,kl->ijl', self.inputs,
                tf.transpose(self.tile))
        vertCompress = tf.divide(vertCompress, self.n)

        # tile out (d x 1 x n) -> (d x n x n)
        vert: tf.Tensor = tf.einsum('ijk,kl->ijl', vertCompress, self.expand)

        # multiply elementwise with input data to force integration
        # TODO: same as above
        out_total: tf.Tensor = vert * self.inputs
        # apply processing to this chunk
        out_total = tf.einsum('ij,ljk->lik', self.W[1], out_total)

        # stack in and out
        # thinking in terms of (d x n^2):
        # (d x n^2) \
        # ---------  > (2d x n^2)
        # (d x n^2) /
        output: tf.Tensor = tf.concat([in_total, out_total], 1)
        
        # apply last weight matrix to reduce dimensionality to (d x n^2)
        output = tf.einsum('ij,kjl->kil', self.W[2], output)

        # tile out (d x 1) bias matrix to (d x n^2) and add to outputs
        # disregarding batching of course
        post_bias: tf.Tensor = tf.add(output,
                tf.tile(self.B, [self.batchsize, 1, self.n * self.n]))

        # return biased outputs after activation
        return self.activation(post_bias)

    def output_shape(self) -> Tuple[int, int, int]:
        return self.input_shape

    @classmethod
    def _import_default_params(cls) -> object:
        from ..conf import locality1
        return locality1

