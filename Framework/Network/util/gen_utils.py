from typing import Dict
import numpy as np
import tensorflow as tf
"""Helper ops for building networks."""

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
