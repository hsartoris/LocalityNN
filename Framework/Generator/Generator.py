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

    def __init__(self, module: Callable, num_neurons: int) -> None:
        self._module: AbstractGenerator = module(num_neurons)
   
    def get_default_params(self) -> Dict:
        return self._module._get_default_params()

    def get_params(self) -> Dict:
        return self._module.get_params()

    def set_params(self, params: Dict) -> None:
        self._module.set_params(params)

    def new_structure(self) -> np.ndarray:
        return self._module.new_structure()

    def get_structure(self) -> np.ndarray:
        return self._module.get_structure()

