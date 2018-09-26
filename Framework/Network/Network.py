from .NetworkModule import NetworkModule
from .Stack import Stack
import tensorflow as tf
from typing import List, Dict, Tuple
import os

class Network(NetworkModule):
    """Builds, trains, and runs a complete network.

    Parameters:
        batchsize: int
        init_learn_rate: float
        optimizer
        train_steps: int
        save_dir: str
        epoch_len: int [default: int(num_samples/batchsize)]
        save_epoch: int [default: 1]
        save_step: int (optional)

        data_dir: str
        train_split: float - decimal percentage of total data to allocate to 
            training/validation. remainder goes to testing. [default: 1.0]
        valid_split: float - decimal percentage of remaining data going to 
            validation. [default: .2]

        stack: dict - dict defining the stack

    """

    def _setup(self) -> None:

        self.save_dir: str = self.params['save_dir']
        self.batchsize: int = self.params['batchsize']
        self.init_lr: float = self.params['init_learn_rate']

        self.stack: Stack


    def generate_config(self, indent_level: int = 0) -> str:
        conf: str = "network_params = {\n"
        idl: int = 1
        sp: int = indent_level * 4
        conf: str = self._generate_config(indent_level, skip = "stack")

        conf += " "*sp + "'stack': {\n"
        conf += self.stack.generate_config(idl+1)
        conf += " "*sp + "}\n"


    @classmethod
    def _import_default_params(cls) -> object:
        from ..conf import network
        return network

