from .AbstractLayer import AbstractLayer
from typing import Tuple, Dict, List
import tensorflow as tf

class Stack(AbstractLayer):
    """Class used for concatenating multiple layers into a stack.

    Treated as a layer itself by parent classes.
    """

    def add_layer(self, layer_conf: Tuple) -> None:
        """Creates new layer and adds to stack.
        """
        #TODO: this is all badly typed

        if self.debug: print("Adding layer", len(self.layers))
        if self.debug: print("Config:\n", layer_conf)

        # is it a 3-tuple?
        assert(len(layer_conf) == 3)

        # is the first item a layer module? if so, store it for convenience
        #TODO: figure out how to do this
        #assert(isinstance(layer_conf[0], AbstractLayer))
        layer_module: AbstractLayer = layer_conf[0]

        # do we have a name provided? if so, store it. otherwise, use mod name
        layer_name: str
        if layer_conf[1] is None:
            layer_name = layer_module.__name__
            if self.debug: print("No layer name provided; using class name.")
        else:
            assert(isinstance(layer_conf[1], str))
            layer_name = layer_conf[1]

        # is the third item a dict?
        assert(isinstance(layer_conf[2], dict))
        layer_conf_dict: dict = layer_conf[2]

        if self.debug:
            print("Layer name:", layer_name, "\nLayer module:", 
                    layer_module.__name__)

        with tf.variable_scope(layer_name):
            self.layers.append(layer_module(params = layer_conf_dict,
                                            parent_params = self.params))

        if hasattr(self.layers[-1], "name"):
            print("Warning: layer already has 'name' attribute. Overwriting.")

        self.layers[-1].name: str = layer_name

    @classmethod
    def _import_default_params(cls) -> object:
        from .conf import stack
        return stack

    def _setup(self):
        assert(isinstance(self.params['debug'], bool))
        self.debug: bool = self.params['debug']

        self.layers: List[AbstractLayer] = []
        for layer_conf in self.params['layers']:
            if not isinstance(layer_conf, tuple):
                raise Exception("""Layer configs for Stack should be a 
                Tuple[AbstractLayer, str, Dict[str, any]] of the form (<module>, 
                <name>, <layer params>.""")

            # just in case mypy is picky
            assert(isinstance(layer_conf, tuple))
            self.add_layer(layer_conf)

    def compute_layer_ops(self) -> tf.Tensor:
        pass
