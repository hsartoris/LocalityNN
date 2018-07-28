from ..common import AbstractBase
import numpy as np
from typing import Dict

class AbstractGenerator(AbstractBase):
    """Abstract class for generator modules found in Generator.modules.

    Structures come in the form of adjacency matrices stored in numpy arrays.  

    Attributes:
        new_structure: generates and returns new matrix.
        get_structure: returns previously created matrix.
    """

    def __init__(self, num_neurons, params: Dict = None) -> None:
        """Allocates new matrix of num_neurons x num_neurons."""
        # call to AbstractBase __init__
        super(AbstractGenerator, self).__init__(params)

        # generator specific values
        self.n: int = num_neurons
        self.matrix: np.ndarray = self._generate_structure()

    def _generate_structure(self) -> np.ndarray:
        raise NotImplementedError("""_generate_structure is a method private to 
            modules and should not be called directly.""")

    def new_structure(self) -> np.ndarray:
        self.matrix = self._generate_structure()
        return self.matrix

    def get_structure(self) -> np.ndarray:
        return self.matrix
