from ..tf_names import *
from typing import Dict, List, Tuple

param_defaults: Dict[str, any] = {
        'batchsize': None,
        'data_shape': None,
        'supervisor_params': None,
        'data_dir': None,
        'save_dir': None,
        'train_item_count': None,
        'valid_item_count': None,
        'test_item_count': None,
        'batches_per_epoch': None,
        'epochs': None,
        'epochs_to_save': 5
        }

param_types: Dict[str, any] = {
        'batchsize': int,
        'data_shape': tuple,
        'supervisor_params': dict,
        'data_dir': str,
        'save_dir': str,
        'train_item_count': int,
        'valid_item_count': int,
        'test_item_count': int,
        'batches_per_epoch': int,
        'epochs': int,
        'epochs_to_save': int
        }

requirements: List[str] = [
        'batchsize',
        'data_shape',
        'supervisor_params',
        'data_dir',
        'save_dir',
        'train_item_count',
        'valid_item_count',
        'test_item_count',
        'batches_per_epoch',
        'epochs'
        ]
