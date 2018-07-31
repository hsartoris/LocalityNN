from ..Generator import Generator
import numpy as np

class Barabasi_Albert(Generator):
    """Barabasi-Albert graph generator.

    """

    @classmethod
    def _get_default_params(cls):
        return {p: .1, m0: 2}

    def __init__(self, num_neurons):
        super(num_neurons)

    def _generate_structure(self) -> np.ndarray:
        # TODO
        raise NotImplementedError
