from ..common import Parameterizable
import json
import tensorflow as tf
from typing import Dict

def NetworkBuilder(Parameterizable):
    """Constructs a network based on the parameters specified in the provided 
    JSON file.

    Parameters in the config file are applied to the parameter dictionary for 
    NetworkBuilder, except in the case of the key 'layers'.

    Example config file:

    {
        "batchsize" : 1,
        "inputs_dims" : [ 0 : 4, 1 : 3 ],
        "layers" : [
            {
                "name" : "layername", # used for scoping
                "module" : "Locality0",
                "d" : 3, # module-specific parameter
                "activation" : "tf.nn.relu"
            }
        ]
    }

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
        with open(conf_fname) as f:
            self.json: Dict[str, any] = json.load(f)

        # reference stuff, not functional code
        layers = self.json.pop('layers')

    @classmethod
    def _get_default_params(cls) -> Dict[str, any]:
        return {
                "batchsize" : 1,
                "inputs_dims" : None,
                "train_steps": None,
                "validate_steps" : None,
                "save_ckpt" : False,
                "save_dir" : None,
                "debug" : True
                }
