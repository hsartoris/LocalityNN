import numpy as np
from typing import Dict

class AbstractBase(object):
    """Abstract base for Simulator and Generator templates.

    `params` defines details for simulation that may be specific to a given 
    module.

    Attributes:
        get_params: returns dict containing possible parameters for module
        set_params: accepts dict setting some or all parameters for module
    """

    def __init__(self, params: Dict[str,any] = None) -> None:
        self.params: Dict[str,any] = self._get_default_params()
        if not params is None:
            self.set_params(params)

    @classmethod
    def _get_default_params(cls) -> Dict[str,any]:
        """Class method returning default parameters for given module."""
        raise NotImplementedError("""This implementation is abstract. Call on a 
           subclass instance with get_default_params""")

    def get_params(self) -> Dict[str,any]:
        return self.params

    def set_params(self, params: Dict[str,any]) -> None:
        for key in list(params):
            if not key in list(self.params):
                raise KeyError("Key " + str(key) + " is not a parameter for " + 
                        type(self).__name__)
            else:
                self.params[key] = params[key]
