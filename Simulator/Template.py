import numpy as np
from typing import Dict

class Template(object):
    """Template for simulator modules found in Simulator.modules.

    Accepts structures in the form of numpy adjacency matrices.

    `params` defines details for simulation that may be specific to a given 
    module.

    """

    def __init__(self, params: Dict = None) -> None:
        self.params: Dict = self._get_default_params()
        if not params is None:
            self.set_params(params)

    @classmethod
    def _get_default_params(cls) -> Dict:
        raise NotImplementedError


