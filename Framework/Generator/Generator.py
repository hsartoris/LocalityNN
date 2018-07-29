from .AbstractGenerator import AbstractGenerator
from typing import Callable, Dict
import numpy as np

class Generator(AbstractGenerator):
    """Proxy class for generator modules found in Generator.modules.

    Effectively provides a public interface to modules.

    Example usage:
    gen = Generator(Erdos_Renyi, 5)

    This creates an Erdos-Renyi network generator producing a 5-node matrix.
    """

    def __init__(self, module: Callable, num_neurons: int,
            params: Dict = None) -> None:
        """Creates instance of provided generator module, to which all function 
        calls are passed."""
        self._module: AbstractGenerator = module(num_neurons)
   
    def get_default_params(self) -> Dict:
        """Retrieves default parameters for current module. Implemented on 
        module."""
        return self._module._get_default_params()

    def get_params(self) -> Dict:
        """Retrieves current module parameters. Implemented on AbstractBase."""
        return self._module.get_params()

    def set_params(self, params: Dict) -> None:
        """Allows changing of generator parameters. Implemented on 
        AbstractBase."""
        self._module.set_params(params)

    def new_structure(self) -> np.ndarray:
        """Creates, stores, and returns new adjacency matrix. Implemented on 
        AbstractGenerator. Depends on _generate_structure, which is implemented 
        on the module."""
        return self._module.new_structure()

    def get_structure(self) -> np.ndarray:
        """Returns most recent adjacency matrix. Implemented on 
        AbstractGenerator."""
        return self._module.get_structure()

