import numpy as np

class Generator(object):
    """Public interface to generator modules found in Generator.modules.

    Structures come in the form of adjacency matrices stored in numpy arrays.  
    All methods should be overridden on inheriting subclasses; except for 
    __init__, this class is primarily for documentation.

    params defines details for generation that may be specific to a given 
    module.

    Attributes:
        get_params: returns dict containing possible parameters for module
        set_params: accepts dict setting some or all parameters for module
        new_structure: generates and returns new matrix.
        get_structure: returns previously created matrix, if existant.
    """

    def __init__(self, num_neurons):
        """Allocates new matrix of num_neurons x num_neurons.
        Call from subclass.
        """
        
        self.n = num_neurons
        self.matrix = np.zeros((self.n, self.n))
        self.params = self._get_default_params()

    @classmethod
    def _get_default_params(cls):
        """Class method returning default parameters for given module."""
        raise NotImplementedError

    def _generate_structure(self):
        raise NotImplementedError

    def get_params(self):
        raise NotImplementedError

    def set_params(self, params):
        raise NotImplementedError

    def new_structure(self):
        self.matrix = self._generate_structure()
        return self.matrix

    def get_structure(self):
        return self.matrix
