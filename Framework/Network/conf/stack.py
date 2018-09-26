from typing import Dict, List

param_defaults: Dict[str, any] = {
        'batchsize': None,
        'layers': None,
        'debug': True
        }

param_types: Dict[str, type] = {
        'batchsize': int,
        'layers': list,
        'debug': bool
        }

requirements: List[str] = [
        'batchsize',
        'layers',
        ]
