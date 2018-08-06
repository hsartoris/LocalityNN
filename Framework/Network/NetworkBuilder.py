from importlib import import_module
from .Layer import Layer
import tensorflow as tf
from typing import Dict, List
#import layers as layer_modules

confdir: str = "conf/"

class NetworkBuilder(object):
    """Constructs a network based on the parameters specified in the provided 
    JSON file.

    Parameters in the config file are applied to the parameter dictionary for 
    NetworkBuilder, except in the case of the key 'layers'.

    Example config file:

    {
        "batchsize" : 1,
        "inputs_dims" : [ 4, 3 ],
        "layers" : [
            {
                "name" : "layername", # used for scoping
                "module" : "Locality0",
                "d" : 3, # module-specific parameter
                "activation" : "tf.nn.relu"
            }
        ]
    }

    # TODO: expand if necessary
    Parameters:
        batchsize: int - default 1
        inputs_dims: Tuple[int, int, ...] - must be provided
        train_steps: int - steps to train for, must be provided
        validate_steps: int - increment to run validation at. can be
            interpolated, in which case will be min(500, train_steps/10)
        save_ckpt: bool - default False
        save_dir: str - if save_ckpt and save_dir is blank, will produce a 
            timestamp to create directory
        debug: bool - default True, for now anyway
            
    """

    def __init__(self, conf_fname: str) -> None:
        self.debug: bool = True
        self.log("Initializing network builder")
        self.cwd: str = __file__.rsplit("/", 1)[0] + "/"

        self.json_decoder: JSONDecoder = JSONDecoder()

        self.log("Getting default settings")

        config = self.load_defaults(type(self).__name__)
        
        self.defaults: Dict[str, any] = config['defaults']
        self.types: Dict[str, type] = config['types']
        self.requirements: List[str] = config['requirements']

        with open(conf_fname) as f:
            self.json: Dict[str, any] = self.json_decoder.load_json(f)

        self.log(self.json)

        self.valid_config(self.requirements, self.json, type(self).__name__)
        # if we get here, all NetworkBuilder parameters are satisfied

        self.layers: List[Layer] = self.assemble_layers(self.json.pop('layers'))

    def load_defaults(self, modname: str) -> Dict[str, any]:
        with open(self.cwd + confdir + modname + ".json") as f:
            params: Dict[str, any] = self.json_decoder.load_json(f)
        return params

    def assemble_layers(self, layers_conf: List[Dict[str, any]],
            layers_path: str = ".layers") -> List[Layer]:
        """Interprets 'layers' section of JSON configuration file to create the 
        layer stack.
        """
        # TODO: this is pretty bad typing
        # import directory containing layer modules as a module

        layers: List[Layer] = []
        for layer in layers_conf:
            if "module" not in layer:
                raise AttributeError('"module" must be defined on each layer')

            
            assert(type(layer['module']) == str)
            # remove module name from parameters dict; everything else goes to 
            # module
            mod_name: str = layer.pop('module')
            self.log("Importing module " + mod_name + " from layers")

            # import module from layers. builtin error handling is fine
            module: AbstractLayer = getattr(layer_modules, mod_name)

            
            

    def valid_config(self, reqs: List[str], conf: Dict[str, any],
            name: str) -> None:
        """Scans provided JSON config to ensure all required fields are 
        provided."""
        for key in reqs:
            if not key in conf:
                raise AttributeError(name + " requires a missing parameter, " +
                    key + ". For reference, required parameters: " + str(reqs))
    #
    #    def required_params(self) -> List[str]:
    #        # TODO: expand if necessary
    #        return ["inputs_dims", "train_steps", "layers"]

    #@classmethod
    #def _get_default_params(cls) -> Dict[str, any]:
    #    # TODO: expand if necessary
    #    return {
    #            "batchsize" : (1, int),
    #            "inputs_dims" : (None, Tuple[int, int]),
    #            "train_steps": (None, int),
    #            "validate_steps" : (None, int),
    #            "save_ckpt" : (False, bool),
    #            "save_dir" : (None, str),
    #            "debug" : (True, bool)
    #            }

    def log(self, msg: str) -> None:
        # TODO: convert to logging proper
        if self.debug: print(msg)
