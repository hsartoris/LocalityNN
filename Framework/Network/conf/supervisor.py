from ..tf_names import *
from typing import Dict, List, Tuple

param_defaults: Dict[str, any] = {
        'batchsize': None,
        'data_shape': None,
        'stack_params': None,
        'init_learn_rate': None,
        'optimizer': Adam,
        'loss_op': modified_mse,
        'prediction_activation': softmax,
        'data_dir': None,
        'save_dir': None,
        'shuffle_buffer_size': None,
        'prefetch_buffer': 1,
        'dataset_num_parallel': 2,
        }

param_types: Dict[str, type] = {
        'batchsize': int,
        'data_shape': tuple,
        'stack_params': dict,
        'init_learn_rate': float,
        'optimizer': Optimizer,
        'loss_op': Callable,
        'prediction_activation': Callable,
        'data_dir': str,
        'save_dir': str,
        'shuffle_buffer_size': int,
        'prefetch_buffer': int,
        'dataset_num_parallel': int,
        }

requirements: List[str] = [
        'batchsize',
        'data_shape',
        'stack_params',
        'init_learn_rate',
        'loss_op',
        'data_dir',
        'save_dir'
        ]
