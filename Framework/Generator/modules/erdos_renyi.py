from ..AbstractGenerator import AbstractGenerator
from typing import Dict
import numpy as np

class Erdos_Renyi(AbstractGenerator):
    """Erdos-Renyi graph generator. Uses G(n,p) model.
    
    ###TODO: G(n,p) may be unsatisfactory for small n. exchange models based on 
        n?

    # TODO: documentation in Generator.py
    For method documentation see AbstractGenerator.py

    Parameters
       p: probability of given edge connection. default .1
       self_loop: whether or not to include self loops. default false
    """

    @classmethod
    def _get_default_params(cls) -> Dict:
        return {'p': .1, 'self_loop': False}

    def _generate_structure(self) -> np.ndarray:
        # TODO: mypy doesn't know what type np.random.choice returns
        mat: np.ndarray = np.random.choice([1,0], size=(self.n, self.n),
                p=[self.params['p'], 1-self.params['p']])
        if not self.params['self_loop']:
            for i in range(self.n):
                mat[i,i] = 0
        return mat


