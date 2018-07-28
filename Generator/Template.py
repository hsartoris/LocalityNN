import numpy as np
from typing import Dict

class Template(object):
    """Template for generator modules found in Generator.modules.

    Structures come in the form of adjacency matrices stored in numpy arrays.  

    `params` defines details for generation that may be specific to a given 
    module.

    Attributes:
        get_params: returns dict containing possible parameters for module
        set_params: accepts dict setting some or all parameters for module
        new_structure: generates and returns new matrix.
        get_structure: returns previously created matrix.
    """

    def __init__(self, num_neurons, params: Dict = None) -> None:
        """Allocates new matrix of num_neurons x num_neurons.
        Call from subclass.
        """
        self.n: int = num_neurons
        self.params: Dict = self._get_default_params()
        if not params is None:
            self.set_params(params)
        self.matrix: np.ndarray = self._generate_structure()

    @classmethod
    def _get_default_params(cls) -> Dict:
        """Class method returning default parameters for given module."""
        raise NotImplementedError("""_get_default_params is a class method 
            implemented on individual modules, not Template or Generator. For 
            getting params from an instantiated Generator, use 
            get_default_params""")

    def _generate_structure(self) -> np.ndarray:
        raise NotImplementedError("""_generate_structure is a method private to 
            modules and should not be called directly.""")

    def get_params(self) -> Dict:
        return self.params

    def set_params(self, params: Dict) -> None:
        for key in list(params):
            if not key in list(self.params):
                raise KeyError("Key " + str(key) + " is not a parameter on " + 
                        type(self).__name__)
            else:
                self.params[key] = params[key]

    def new_structure(self) -> np.ndarray:
        self.matrix = self._generate_structure()
        return self.matrix

    def get_structure(self) -> np.ndarray:
        return self.matrix
