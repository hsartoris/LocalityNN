from .AbstractSimulator import AbstractSimulator
from typing import Callable, Dict
import numpy as np

class Simulator(AbstractSimulator):
    """Proxy class for simulator modules found in Simulator.modules.

    Effectively creates a public interface to modules.

    Example usage:
    sim = Simulator(matmul, matrix)

    This creates a matrix multiplication simulator for `matrix`.
    """

    def __init__(self, module: Callable, matrix: np.ndarray,
            params: Dict = None) -> None:
        """Creates instance of provided simulator module, to which all function 
        calls are passed."""
        self._module: AbstractSimulator = module(matrix, params)

    def get_default_params(self) -> Dict:
        """Retrieves default parameters for current module. Implemented on 
        module."""
        return self._module._get_default_params()

    def get_params(self) -> Dict:
        """Retrieves current module parameters. Implemented on AbstractBase."""
        return self._module.get_params()

    def set_params(self, params: Dict) -> None:
        """Allows changing of simulator parameters. AbstractSimulator overrides 
        AbstractBase to add a call to _setup, in case values need to be 
        recalculated."""
        self._module.set_params(params)

    def step(self) -> np.ndarray:
        """Calculates a single step forward in time in the simulation.  
        Implemented on module."""
        return self._module.step()

    def n_steps(self, n: int) -> np.ndarray:
        """Calculates n steps forward in time. Provided by AbstractSimulator and 
        depends on step; likely should not be overridden."""
        return self._module.n_steps(n)
