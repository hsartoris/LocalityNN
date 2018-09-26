from .AbstractLayer import AbstractLayer
from typing import Tuple, Dict, List
from .tf_names import *
import tensorflow as tf

class Stack(AbstractLayer):
    """Class used for concatenating multiple layers into a stack.

    Treated as a layer itself by parent classes.
    """

    def add_layer(self,
            layer_conf: Tuple) -> Tuple[AbstractLayer, str, Dict[str, any]]:
        """Creates new layer and adds to stack.
        """
        #TODO: this is all badly typed

        if self.debug: print("Adding layer", len(self.layers))
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

        # get layer inputs
        layer_inputs: tf.Tensor
        if len(self.layers) == 0:
            if self.debug:
                print("Layer 0 inputs: using Stack inputs")
            # first layer
            layer_inputs = self.inputs
        else:
            if self.debug:
                print("Layer " + str(len(self.layers)) +
                        " inputs derived from previous layer.")
            # subsequent layers
            layer_inputs = self.layers[-1].outputs

        if self.debug:
            print("Shape: " + str(layer_inputs.get_shape().as_list()))


        # TODO: some kinda checking here
        layer_conf_dict['input_shape'] = \
            tuple(layer_inputs.get_shape().as_list())

        with tf.variable_scope(layer_name, reuse = tf.AUTO_REUSE):
            self.layers.append(layer_module(layer_inputs,
                                            params = layer_conf_dict,
                                            parent_params = self.params))

        if hasattr(self.layers[-1], "name"):
            print("Warning: layer already has 'name' attribute. Overwriting.")

        self.layers[-1].name: str = layer_name

        return (layer_module, layer_name, self.layers[-1].params)

    @classmethod
    def _import_default_params(cls) -> object:
        from .conf import stack
        return stack

    def _setup(self):
        assert(isinstance(self.params['debug'], bool))
        self.debug: bool = self.params['debug']

        self.layers: List[AbstractLayer] = []
        for i in range(len(self.params['layers'])):
            layer_conf = self.params['layers'][i]
            layer_tup: tuple
            if not isinstance(layer_conf, tuple):
                # assume it's just a module
                layer_tup = (layer_conf, None, dict())
            elif len(layer_conf) == 2:
                # skipped either dict or name
                if isinstance(layer_conf[1], dict):
                    # skipped name
                    layer_tup = (layer_conf[0], None, layer_conf[1])
                else:
                    # skipped dict
                    layer_tup = (layer_conf[0], layer_conf[1], dict())
            else:
                # complete config tuple
                layer_tup = layer_conf
            self.params['layers'][i] = self.add_layer(layer_tup)


    def output_shape(self) -> Tuple[int, int, int]:
        return tuple(self.layers[-1].get_shape().as_list())

    def _compute_layer_ops(self) -> tf.Tensor:
        return self.layers[-1].outputs

    def generate_config(self, indent_level: int = 0) -> str:
        """Generates a config file for creating a network from above the 
        Framework directory.
        """
        idl: int = indent_level
        sp: int = indent_level * 4
        conf: str = self._generate_config(indent_level, skip = "layers")

        conf += " "*sp + "'layers': [\n"
        for layer in self.layers:
            conf += " "*(idl+1)*4 + "(" + type(layer).__name__ + ", "
            conf += "'" + layer.name + "',\n"
            conf += " "*(idl+2)*4 + "{\n"
            conf += layer.generate_config(idl+3)
            #for key in layer[2]:
            #    conf += " "*16 + "'" + key + "': " + get_str(layer[2][key])
            #    conf += ",\n"
            conf += " "*(idl+2)*4 + "}\n"
            conf += " "*(idl+1)*4 + "),\n"

        conf += " "*sp + "]\n"
        return conf



