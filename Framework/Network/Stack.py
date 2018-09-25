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

        # get layer inputs
        layer_inputs: tf.Tensor
        if len(self.layers) == 0:
            # first layer
            layer_inputs = self.inputs
        else:
            # subsequent layers
            layer_inputs = self.layers[-1].outputs

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
            if not isinstance(layer_conf, tuple):
                raise Exception("""Layer configs for Stack should be a 
                Tuple[AbstractLayer, str, Dict[str, any]] of the form (<module>, 
                <name>, <layer params>.""")

            # just in case mypy is picky
            assert(isinstance(layer_conf, tuple))
            self.params['layers'][i] = self.add_layer(layer_conf)

    def _compute_layer_ops(self) -> tf.Tensor:
        return self.layers[-1].outputs

    def generate_config(self) -> str:
        print(names)
        def get_str(conf_item) -> str:
            if hasattr(conf_item, "real_dtype"):
                #tf dtypes
                return "tf." + conf_item.name
            if conf_item in names:
                return names[conf_item]
            if hasattr(conf_item, "__name__"):
                return conf_item.__name__
            return str(conf_item)

        """Generates a config file for creating a network from above the 
        Framework directory.
        """
        conf = "from Framework.Network.layers import *\n"
        conf += "from Framework.Network.tf_names import *\n\n"
        conf += "params = {\n"

        for key in self.params:
            # deal with the layers at the end
            if key == "layers": continue

            conf += "    '" + key + "': " + get_str(self.params[key]) + ",\n"

        conf += "    'layers': [\n"
        for layer in self.params['layers']:
            conf += " "*8 + "(" + get_str(layer[0]) + ", "
            conf += "'" + layer[1] + "',\n"
            conf += " "*12 + "{\n"
            for key in layer[2]:
                conf += " "*16 + "'" + key + "': " + get_str(layer[2][key])
                conf += ",\n"
            conf += " "*12 + "}\n"
            conf += " "*8 + "),\n"

        conf += " "*4 + "]\n"
        conf += "}\n"
        return conf



