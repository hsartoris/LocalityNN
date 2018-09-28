from typing import Dict
import numpy as np
import tensorflow as tf
import importlib.util
import os.path
import time
import sys
from ..util import Log as log

"""Helper ops for building and running networks."""

def make_expand(n: int, dtype: type = np.float32) -> np.ndarray:
    """Creates constant expand matrix"
    
    `expand` appears as such for n == 3:
        | 1 1 1 0 0 0 0 0 0 |
        | 0 0 0 1 1 1 0 0 0 |
        | 0 0 0 0 0 0 1 1 1 |
    """
    expand: np.ndarray = np.array([[1]*n + [0]*n*(n-1)])
    for i in range(1, n):
        expand = np.append(expand, [[0]*n*i + [1]*n + [0]*n*(n-1-i)], 0)
    return expand


def import_from_path(path: str, mod_name: str = None) -> object:
    """imports and returns a specified module.

    If mod_name is not provided, assumes that the last element of path is of the 
    format <mod_name>.py and attempts to parse accordingly.
    """
    # TODO: return type

    if mod_name is None:
        fname: str = os.path.split(path)[1]
        if len(fname) < 4 or '.' not in fname:
            # filename has to be at least <some char>.py
            log.error("""import_from_path: no module name provided, and given 
                path (""" + path + ") does not contain a valid filename")
            sys.exit()
        mod_name = fname.split('.')[0]

    spec = importlib.util.spec_from_file_location(mod_name, path)
    # TODO: type
    module: object = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

