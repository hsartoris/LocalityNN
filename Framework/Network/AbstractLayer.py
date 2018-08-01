from ..common import AbstractBase
import tensorflow as tf
from typing import List, Callable

def Layer(AbstractBase):
    """Abstract class for wrapping layer ops into a full layer.

    Extends AbstractBase for parameter management.

    Child classes can optionally override _setup, and must override _layer_ops.

    Any and all calls to create variables in these methods should use 
    tf.get_variable, to maintain name scoping.
    
    """

    def __init__(self, inputs: tf.Tensor, activation: Callable,
            name: str = None, batched: bool = True, 
            params: Dict = None) -> None:
        # call superclass initializer with params as argument
        super(Layer, self).__init__(params)

        # record provided information. not everything will be used by every 
        # layer implementation, but all values should be generally useful
        self.inputs: tf.Tensor = inputs
        self.input_shape: List = inputs.get_shape.as_list()
        self.dtype: type = inputs.dtype
        self.activation: Callable = activation
        self.batched: bool = batched
        
        # TODO: convert this to a logging message
        if name is None:
            self.use_scope: bool = False
            print("Warning: this instance of " + type(self).__name__ +
                    " is running unscoped.")
        else:
            self.use_scope: bool = True
            self.name: str = name

        if self.use_scope:
            with tf.variable_scope(self.name):
                self._setup()
                self.outputs: tf.Tensor = self._layer_ops()
        else:
            self._setup()
            self.outputs: tf.Tensor = self._layer_ops()

    def _setup(self) -> None:
        """Any operations that should be run before _layer_ops. Optional.

        Override on subclasses.
        """
        pass


    def _layer_ops(self, inputs: tf.Tensor) -> tf.Tensor:
        """Responsible for actual layer computations.

        Must be overridden by child classes.
        """
        raise NotImplementedError

    
