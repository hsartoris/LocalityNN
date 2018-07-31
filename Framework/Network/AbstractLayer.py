from ..common import AbstractBase
import tensorflow as tf
from typing import List

def Layer(AbstractBase):
    """Abstract class for wrapping layer ops into a full layer.

    Extends AbstractBase for parameter management.

    Adds __call__(inputs: tf.Tensor) to expedite usage in models.
    
    """

    def __init__(self, inputs: tf.Tensor, name: str = None,
            dtype: type = tf.float32, params: Dict = None) -> None:
        # call superclass initializer with params as argument
        super(Layer, self).__init__(params)

        # record provided information
        self.inputs: tf.Tensor = inputs
        self.input_shape: List = inputs.get_shape.as_list()
        self.dtype: type = dtype
        
        # TODO: convert this to a logging message
        if name is None:
            self.use_scope: bool = False
            print("Warning: this instance of " + type(self).__name__ +
                    " is running unscoped.")
        else:
            self.use_scope: bool = True
            self.name: str = name

        self.outputs: tf.Tensor
        self._layer_ops()



    def _layer_ops(self, inputs: tf.Tensor) -> tf.Tensor:
        """Responsible for actual layer computations.

        Must be overridden by child classes.
        """
        raise NotImplementedError

    
