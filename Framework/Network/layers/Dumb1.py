from ..AbstractLayer import AbstractLayer
from typing import Dict, List, Tuple
import tensorflow as tf

class Dumb1(AbstractLayer):
    """Maps from (a x b x c) to (a x b x c) via a matrix multiplication of
    (b x b) einsummed across the inputs.

    Dumber sibling of Locality1
    """

    def _setup(self) -> None:
        self.W: tf.Tensor = tf.get_variable("weights",
                shape = (self.params['input_shape'][1],
                    self.params['input_shape'][1]),
                dtype = self.params['dtype'],
                initializer = self.params['initializer_w'](
                    stddev = self.params['stddev_w']))

        self.B: tf.Tensor = tf.get_variable("biases",
                shape = (1, self.params['input_shape'][1], 1),
                dtype = self.params['dtype'],
                initializer = self.params['initializer_b'](
                    stddev = self.params['stddev_b']))

    def _compute_layer_ops(self) -> tf.Tensor:
        outputs: tf.Tensor = tf.einsum('ij,ljk->lik', self.W, self.inputs)
        post_bias: tf.Tensor = tf.add(outputs, tf.tile(self.B, 
                [self.params['batchsize'], 1, self.params['input_shape'][2]]))

        return self.params['activation'](post_bias)

    def output_shape(self) -> Tuple[int, int, int]:
        return self.params['input_shape']

    @classmethod
    def _import_default_params(cls) -> object:
        from ..conf import dumb1
        return dumb1

