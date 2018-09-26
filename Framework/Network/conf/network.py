from ..tf_names import *
from typing import Dict, List, Tuple

param_defaults: Dict[str, any] = {
        'batchsize': None,
        'init_learn_rate': None,
        'train_steps': None,
        'save_dir': None,
        'stack': None
        }

param_types: Dict[str, type] = {
        'batchsize': int,
        'init_learn_rate': float,
        'train_steps': int,
        'save_dir': str,
        'stack': dict
        }

requirements: List[str] = []
