from ..tf_names import *
from typing import Dict, List, Tuple

param_defaults: Dict[str, any] = {
        'batchsize': None,
        'data_shape': None,
        'stack_params': None,
        'init_learn_rate': None,
        'optimizer': Adam,
        'loss_op': absolute_difference,
        'prediction_activation': softmax
        }

param_types: Dict[str, type] = {
        'batchsize': int,
        'data_shape': tuple,
        'stack_params': dict,
        'init_learn_rate': float,
        'optimizer': Optimizer,
        'loss_op': Callable,
        'prediction_activation': Callable

        }

requirements: List[str] = [
        'batchsize',
        'data_shape',
        'stack_params',
        'init_learn_rate',
        'loss_op'
        ]
