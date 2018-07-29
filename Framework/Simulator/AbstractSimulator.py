from ..common import AbstractBase
import numpy as np
from typing import Dict

class AbstractSimulator(AbstractBase):
    """Abstract class for simulator modules. Extends AbstractBase with 
    simulator-specific needs.

    See Simulator.py for public function documentation.
    """

    def __init__(self, matrix: np.ndarray, params: Dict = None) -> None:
        # call superclass initializer with params as argument
        super(AbstractSimulator, self).__init__(params)
        
        # store provided matrix locally
        self.matrix: np.ndarray = matrix
        # local setup operations, if present
        self._setup()

    def _setup(self) -> None:
        """Called at the end of __init__ and set_params. Allows modules to 
        perform unique setup operations. Optional."""
        pass

    def step(self) -> np.ndarray:
        raise NotImplementedError

    def n_steps(self, n: int) -> np.ndarray:
        out: np.ndarray = self.step()
        for i in range(1, n):
            out = np.concatenate((out, self.step()), axis = 1)
        return out

    def set_params(self, params: Dict) -> None:
        """Overrides AbstractBase set_params method to allow modules to 
        reperform setup operations after parameter change."""
        super(AbstractSimulator, self).set_params(params)
        self._setup()
