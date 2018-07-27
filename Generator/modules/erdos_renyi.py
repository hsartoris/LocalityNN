from ..Generator import Generator
import numpy as np

class Erdos_Renyi(Generator):
    """Erdos-Renyi graph generator. Uses G(n,p) model.

    For method documentation see Generator.py

    Parameters
       p: probability of given edge connection. default .1
       self_loop: whether or not to include self loops. default false
    """

    @classmethod
    def _get_default_params(cls):
        return {p: .1, self_loop: False}

    def __init__(self, num_neurons):
        super(num_neurons)

    def _generate_structure(self):
        mat = np.random.choice([1,0], size=(self.n, self.n),
                p=[self.params['p'], 1-self.params['p']])
        if not self.params['self_loop']:
            for i in range(self.n):
                mat[i,i] = 0
        return mat


