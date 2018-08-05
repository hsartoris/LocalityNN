import json
import tensorflow as tf
from typing import Union, Dict, List, Callable
from typing.io import TextIO
"""Support modules for JSON management."""

activation_type_enc: str = "Callable[[tf.Tensor, str], tf.Tensor]"
activation_type: type = Callable[[tf.Tensor, str], tf.Tensor]

initializer_type: type = tf.keras.initializers.Initializer

# TODO: write activations
initializers_conf_path = "../conf/initializers.json"
activations_conf_path = "../conf/activations.json"

class ArrayEncoder(json.JSONEncoder):
    """Layer over default JSON encoder to provide Tuple storage in conjunction 
    with hinted_tuple_hook.
    
    Thanks to: 
    https://stackoverflow.com/questions/15721363/preserve-python-tuples-with-json
    """
    def encode(self, obj: object) -> str:
        # TODO: I don't think this typing can be improved but I wish it could
        def hint_stuff(item: object) -> object:
            """Reshapes input data to preserve tuples.

            Also replaces Tensorflow modules with name.
            """
            if isinstance(item, tuple):
                return {"__tuple__" : True, "items" : item}
            if isinstance(item, list):
                return [hint_stuff(i) for i in item]
            if isinstance(item, dict):
                return {key : hint_stuff(value) for key, value in item.items()}

            if item == activation_type:
                # activation function types are awful
                return activation_type_enc
            # testing to see what exactly can and can't be encoded
            #if isinstance(item, Callable):
            #    # in general we'll represent Callables with their names
            #    return item.__name__
            return item

        return super(ArrayEncoder, self).encode(hint_stuff(obj))

class JSONDecoder(object):
    """Class to assist decoding JSON configs into usable information.

    I've decided to handle actual module loading problems in NetworkBuilder
    """
    def __init__(self) -> None:
        cwd: str= __file__.rsplit("/", 1)[0] + "/"
        with open(cwd + initializers_conf_path) as f:
            self.initializers: Dict[str, str] = json.load(f)
        #with open(activations_conf_path) as f:
        #    self.activations: Dict[str, str] = json.load(f)


    def json_decode_hook(self, obj: dict) -> Union[dict, tuple]:
        # TODO: this needs to handle activation_type and general Callables
        # apparently the object hook is only called for dicts?
        if "__tuple__" in obj:
            return tuple(obj['items'])
        for key, value in obj.items():
            if value == activation_type_enc:
                # activation type; replace with actual
                obj[key] = activation_type
            elif value == initializer_type.__name__:
                obj[key] = initializer_type
        return obj
    
    def load_json(self, json_file: TextIO) -> Union[dict, list]:
        # TODO: allow for loading from files and strings
        #   this would be easy but I don't want to do it
        return json.load(json_file, object_hook = self.json_decode_hook)

def encode_json(inputs: Union[dict, list]) -> str:
    enc: ArrayEncoder = ArrayEncoder()
    return enc.encode(inputs)

