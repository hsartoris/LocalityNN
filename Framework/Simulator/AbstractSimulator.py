from ..common import AbstractBase
import numpy as np
from typing import Dict

class AbstractSimulator(AbstractBase):
    """Abstract class for simulator modules. Extends AbstractBase with 
    simulator-specific needs.
    
    """

    def __init__(self, matrix: np.ndarray, params: Dict = None) -> None:
        super(AbstractSimulator, self).__init__(params)
        self.matrix: np.ndarray = matrix
        self._setup()

    def _setup(self) -> None:
        # This function will likely by highly customized to each module

    def step(self) -> np.ndarray:
        raise NotImplementedError

    def n_steps(self, n: int) -> np.ndarray:
        raise NotImplementedError

    def set_params(self, params: Dict) -> None:
        super(AbstractSimulator, self).set_params(params)
        self._setup()
