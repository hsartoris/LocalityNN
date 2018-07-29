from ..AbstractSimulator import AbstractSimulator
from typing import Dict
import numpy as np

class matmul(AbstractSimulator):
    """Matrix multiplication simulator. Simplest it can get.

    Parameters:
        p: independent spike probability
        min: lowest allowed node level
        max: highest allowed node level

    Variables:
        matrix: stored adjacency matrix
        vector: most recent step

    # TODO: implement parameter for non-binary states and weights
    """

    @classmethod
    def _get_default_params(cls) -> Dict:
        return {'p': .2, 'min': 0, 'max': 1}

    def _random_vector(self) -> np.ndarray:
        """Creates binary vector of length n with elements set either to 0 or 1 
        based on probability parameter"""

        return np.random.choice([1,0], size=(self.matrix.shape[0], 1), 
                p=[self.params['p'], 1-self.params['p']])

    def _setup(self) -> None:
        self.vector: np.ndarray = self._random_vector()

    def step(self) -> np.ndarray:
        # multiply vector for next step
        self.vector = np.matmul(self.matrix, self.vector)
        # add random noise to vector and clip to allowed values
        self.vector = np.clip(self.vector + self._random_vector(),
                self.params['min'], self.params['max'])
        return self.vector
