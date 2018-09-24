from .AbstractLayer import AbstractLayer
from .conf import stack

class Stack(AbstractLayer):
    """Class used for concatenating multiple layers into a stack.

    Treated as a layer itself by parent classes.
    """

    @classmethod
    def _import_default_params(cls) -> object:
        return stack

    def setup(self):
        print("improbably actually got here")
