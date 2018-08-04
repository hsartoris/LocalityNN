from .Layer import Layer
import tensorflow as tf
from typing import Dict, List
import configparser

class LayerStack(object):
    """Interprets dict of config information from Network to produce a stack of 
    Layer modules.
    """

    def __init__(self, config: Dict[any]) -> None:
        # TODO: improve typing
        self.layers: List[Layer] = []
