from .NetworkModule import NetworkModule
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

    """

    def _setup(self) -> None:

        self.save_dir: str = self.params['save_dir']
        self.batchsize: int = self.params['batchsize']
        self.init_lr: float = self.params['init_learn_rate']



    @classmethod
    def _import_default_params(cls) -> object:
        from ..conf import network
        return network

