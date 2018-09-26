from .NetworkModule import NetworkModule
from abc import abstractmethod
import tensorflow as tf
from typing import List, Callable, Dict, Tuple

class AbstractLayer(NetworkModule):
    """Abstract class implementing parameterization for layers and stacks.

    """

    def __init__(self, inputs: tf.Tensor, params: Dict[str, any] = None,
            parent_params: Dict[str, any] = None):
        self.inputs: tf.Tensor = inputs
        super().__init__(params, parent_params)

        #self._setup()
        self.outputs: tf.Tensor = self._compute_layer_ops()

    @abstractmethod
    def _compute_layer_ops(self) -> tf.Tensor:
        """Actual layer computations performed here.
        """
        
    @classmethod
    @abstractmethod
    def _import_default_params(cls) -> object:
        """Class method that should import default params for a given module and 
        return them.

        Object returned will be a module containing params_default, params_type, 
        and requirements dicts and lists.
        
        Must be overridden on module.
        """

    @abstractmethod
    def output_shape(self) -> Tuple[int,int,int]:
        """Module reports expected output size given input size.
        """
