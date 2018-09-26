from ..AbstractLayer import AbstractLayer
from typing import Dict, List, Tuple
import tensorflow as tf

class Flatten(AbstractLayer):
    """Flattens an (a x b x c) matrix to (a x 1 x c).

    Parameters:
        input_shape: Tuple[int,int,int] - dimensions of input data
        stddev_w: float - standard deviation of weight matrix
        initializer_w: Initializer - initializer for weight matrix
        activation: Activation - optional final activation
        dtype: type - type of input data
    """

    def _setup(self) -> None:
        """Uses input dimesions to instantiate weight matrix.
        """

        self.W: tf.Tensor = tf.get_variable("weights",
                shape = (1, self.params['input_shape'][1]),
                dtype = self.params['dtype'],
                initializer = self.params['initializer_w'](
                    stddev = self.params['stddev_w']))

        self.activate: bool = self.params['activation'] is not None

    def _compute_layer_ops(self) -> tf.Tensor:
        outputs: tf.Tensor = tf.einsum('ij,ljk->lik', self.W, self.inputs)

        if self.activate:
            return self.params['activation'](outputs)
        else:
            return outputs

    def output_shape(self) -> Tuple[int, int, int]:
        return (self.params['input_shape'][0], 1, self.params['input_shape'][2])

    @classmethod
    def _import_default_params(cls) -> object:
        from ..conf import flatten
        return flatten
