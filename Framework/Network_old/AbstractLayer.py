from abc import ABC, abstractmethod
import tensorflow as tf
from typing import List, Callable, Dict, Tuple
from .util import types

class AbstractLayer(ABC):
    """Abstract class for wrapping layer ops into a full layer.

    Does not use Parameterizable because Tensorflow graphs might as well be 
    immutable.

    Child classes can optionally override _setup, and must override _layer_ops.

    Any and all calls to create variables in these methods should use 
    tf.get_variable, to maintain name scoping.
    
    """

    def __init__(self, inputs: tf.Tensor, params: Dict) -> None:
        # compare provided params against defaults
        self._validate_params(params)

        # save hook to inputs and derive dimensionality from it
        self.inputs: tf.Tensor = inputs
        self.input_shape: List[int] = inputs.get_shape().as_list()
        self.activation: Callable[[tf.Tensor, str], tf.Tensor] = \
                self.params['activation']

        # TODO: probably pass in the type
        self.dtype: type = inputs.dtype

        self.batchsize: int = self.params['batchsize']

        self.name: str = self.params['name']
        
        # TODO: convert this to a logging message
        self.use_scope: bool = self.names is not None

        if not self.use_scope:
            print("Warning: this instance of " + type(self).__name__ +
                    " is running unscoped.")

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

    @classmethod
    def _base_default_params(cls) -> types.params:
        """Return dict of default parameters common to all layers.

        Return type, defined in util/types.py, is Dict[str,Tuple[any,type,bool]]

        Format: { <key> : (<default_value>, <param_type>, <is_required>) }
        """
        return {
                "activation" : (None, types.activation, True),
                "name" : (None, str, False),
                "batchsize" : (None, int, True)
                }

    @classmethod
    @abstractmethod
    def _layer_default_params(cls) -> types.params:
        """Implemented by child class to report its individual default 
        parameters.
        """

    @classmethod
    def get_default_parameters(cls) -> types.params:
        """Public method to get all parameters for layer, including inherited.
        """
        base_params: types.params = cls._base_default_params()
        layer_params: types.params = cls._layer_default_params()
        # combine and return dictionaries
        return {**base_params, **layer_params}



    @abstractmethod
    def _get_global_constants(self) -> None:
        """Optional method to allow modules to retrieve global constants before 
        scoping is used. Must use tf.get_variable.
        """
        pass

    @abstractmethod
    def _setup(self) -> None:
        """Any operations that should be run before _layer_ops. Optional.

        Override on subclasses.
        """
        pass

    @abstractmethod
    def _layer_ops(self) -> tf.Tensor:
        """Responsible for actual layer computations.

        Must be overridden by child classes.
        """
