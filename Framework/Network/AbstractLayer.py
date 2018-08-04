from ..common import Parameterizable
import tensorflow as tf
from typing import List, Callable, Dict, Tuple

class AbstractLayer(Parameterizable):
    """Abstract class for wrapping layer ops into a full layer.

    Extends Parameterizable for parameter management.

    Child classes can optionally override _setup, and must override _layer_ops.

    Any and all calls to create variables in these methods should use 
    tf.get_variable, to maintain name scoping.

    `__init__` does not provide default values. Those are provided by the 
    wrapper class, Layer.
    
    """

    def __init__(self, inputs: tf.Tensor,
            activation: Callable[[tf.Tensor, str], tf.Tensor],
            name: str,
            batchsize: int,
            params: Dict) -> None:
        # call superclass initializer with params as argument
        super(AbstractLayer, self).__init__(params)

        # record provided information. not everything will be used by every 
        # layer implementation, but all values should be generally useful
        self.inputs: tf.Tensor = inputs
        self.input_shape: List[int] = inputs.get_shape().as_list()
        self.dtype: type = inputs.dtype
        self.activation: Callable[[tf.Tensor, str], tf.Tensor] = activation

        self.batchsize: int = batchsize
        
        # TODO: convert this to a logging message
        self.use_scope: bool
        if name is None:
            self.use_scope = False
            print("Warning: this instance of " + type(self).__name__ +
                    " is running unscoped.")
        else:
            self.use_scope = True
            self.name: str = name

        # allow module to retrieve globally scoped constants
        self._get_global_constants()

        # run local operations, with scope if name is provided
        if self.use_scope:
            with tf.variable_scope(self.name):
                self._setup()
                self.outputs: tf.Tensor = self._layer_ops()
        else:
            self._setup()
            self.outputs: tf.Tensor = self._layer_ops()
    
    def _get_global_constants(self) -> None:
        """Optional method to allow modules to retrieve global constants before 
        scoping is used. Must use tf.get_variable.
        """
        pass

    def _setup(self) -> None:
        """Any operations that should be run before _layer_ops. Optional.

        Override on subclasses.
        """
        pass


    def _layer_ops(self) -> tf.Tensor:
        """Responsible for actual layer computations.

        Must be overridden by child classes.
        """
        raise NotImplementedError

    
