from typing import Dict, List

param_defaults: Dict[str, any] = {
        'batchsize': None,
        'layers': None,
        'input_dims': None
        }

param_types: Dict[str, type] = {
        'batchsize': int,
        'layers': list,
        'input_dims': tuple
        }

requirements: List[str] = [
        'batchsize',
        'layers',
        'input_dims'
        ]
