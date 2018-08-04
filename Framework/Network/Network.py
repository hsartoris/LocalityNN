# TODO: work in progress. this may need to be rethought
from Framework.Network import Layer
from Framework.Network.utils import make_expand
import configparser
import tensorflow as tf
from typing import Tuple, List
from .Layer import Layer

class Network(object):
    """Skeleton class that will eventually host networks.
    """
    def __init__(self, configpath: str) -> None:
        self.config: configparser.ConfigParser = configparser.ConfigParser()
        self.config.read(configpath)

        self.get_network_config(self.config['Network'])

    def get_network_config(self, netcfg) -> None:
        self.batchsize: int = int(netcfg['batchsize'])
        self.inputs_dim: Tuple[int, int] = (
            int(netcfg['inputs_dim1']),
            int(netcfg['Network']['inputs_dim2']))

    def get_layer_config(self, config) -> None:
        self.layers: List[Layer] = []
        layers_started: bool = False
        for section in config:
            if not layers_started:
                if section == 'Layers':
                    layers_started = True
                continue
            self.layers.append(self.get_layer(config['section']))

    def get_layer(self, section) -> Layer:


