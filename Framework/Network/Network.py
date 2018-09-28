from .NetworkModule import NetworkModule
from .Stack import Stack
import tensorflow as tf
from typing import List, Dict, Tuple
import os
from .util import parse_tfrecord

class Network(NetworkModule):
    """Builds, trains, and runs a complete network.
    
    Designed to be compatible with the TensorFlow Estimator library.

    Parameters:
        batchsize: int
        init_learn_rate: float
        optimizer
        train_steps: int
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
        self.batchsize: int = self.params['batchsize']
        self.init_lr: float = self.params['init_learn_rate']

        self.data_shape: Tuple[int, int] = self.params['data_shape']

        self.input_shape: Tuple[int, int, int] = (self.batchsize,
                self.data_shape[0], self.data_shape[1])

        self.data_dir: str = self.params['data_dir']
        # TODO: type
        self.dataset_train = load_dataset(
                data_dir = self.params['data_dir'],
                name = "train",
                shuffle_buffer_size = self.params['shuffle_buffer_size'],
                batchsize = self.batchsize)

        self.dataset_validate = load_dataset(
                data_dir = self.params['data_dir'],
                name = "validate",
                shuffle_buffer_size = self.params['shuffle_buffer_size'],
                batchsize = self.batchsize)

        self.inputs: tf.Tensor = tf.placeholder(tf.float32, self.input_shape)

        self.stack: Stack = Stack(self.inputs,
                params = self.params['stack_params'],
                parent_params = self.params)
        self.outputs: tf.Tensor = self.stack.outputs


    def generate_config(self, indent_level: int = 0) -> str:
        conf: str = "from Framework.Network.layers import *\n"
        conf += "from Framework.Network.tf_names import *\n\n"
        conf += "network_params = {\n"
        idl: int = 1
        sp: int = idl * 4
        conf += self._generate_config(idl, skip = "stack_params")

        conf += " "*sp + "'stack_params': {\n"
        conf += self.stack.generate_config(idl+1)
        conf += " "*sp + "}\n"
        conf += "}\n"
        return conf


    @classmethod
    def _import_default_params(cls) -> object:
        from .conf import network
        return network

