from .AbstractLayer import AbstractLayer
import tensorflow as tf
from typing import Callable, Dict

class Layer(AbstractLayer):
    # TODO: more documentation
    """Proxy class for layer modules.

    """

    def __init__(self, module: AbstractLayer,
            inputs: tf.Tensor,
            activation: Callable[[tf.Tensor, str], tf.Tensor],
            name: str = None,
            batchsize: int = 32,
            params: Dict = None) -> None:
        """Initializes layer module with default parameters.

        Note that the parameters specified here are only those global to all 
        layer modules.
        """
        self._module: AbstractLayer = module(inputs,
                                        activation,
                                        name,
                                        batchsize,
                                        params)

    def get_default_params(self) -> Dict[str, any]:
        """Retrieves parameters, including applicable defaults, from the current 
        layer module.

        Implemented on module.
        """
        return self._module._get_default_params()

    def get_params(self) -> Dict[str, any]:
        """Retrieves currend module parameters.

        Implemented on Parameterizable.
        """
        return self._module.get_params()

    def set_params(self, params: Dict[str, any]) -> None:
        """Provide new parameters to layer module. Should only be used to 
        provide parameters nonessential to layer structure, as those 
        calculations will happen at instantiation.

        Parameters necessary at instantiation should be passed to `__init__`.
        """
        self._module.set_params(params)

    def outputs(self) -> tf.Tensor:
        return self._module.outputs
