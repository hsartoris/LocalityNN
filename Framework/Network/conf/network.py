from ..tf_names import *
from typing import Dict, List, Tuple

param_defaults: Dict[str, any] = {
        'batchsize': None,
        'data_shape': None,
        'init_learn_rate': None,
        'train_steps': None,
        'save_dir': None,
        'stack_params': None
        }

param_types: Dict[str, type] = {
        'batchsize': int,
        'data_shape': tuple,
        'init_learn_rate': float,
        'train_steps': int,
        'save_dir': str,
        'stack_params': dict
        }

requirements: List[str] = [
        'batchsize',
        'data_shape'
        ]
