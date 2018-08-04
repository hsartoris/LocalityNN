from typing import Dict
import numpy as np
import tensorflow as tf
"""Helper ops for building networks."""

def make_expand(n: int, dtype: type = np.float32) -> np.ndarray:
    """Creates two constant matrices used in the functioning of this layer.
    
    `expand` appears as such for n == 3:
        | 1 1 1 0 0 0 0 0 0 |
        | 0 0 0 1 1 1 0 0 0 |
        | 0 0 0 0 0 0 1 1 1 |

    `tile` appears as such for n == 3:
        | 1 0 0 1 0 0 1 0 0 |
        | 0 1 0 0 1 0 0 1 0 |
        | 0 0 1 0 0 1 0 0 1 |
    """
    expand: np.ndarray = np.array([[1]*n + [0]*n*(n-1)])
    for i in range(1, n):
        expand = np.append(expand, [[0]*n*i + [1]*n + [0]*n*(n-1-i)], 0)
    return expand
    """
    # TODO: `tile` may not be necessary; does tf.tile work for all cases?
    tile: np.ndarray = np.array([([1] + [0]*(n-1))*n])
    for i in range(1, n):
        tile = np.append(tile, [([0]*i + [1] + [0]*(n-1-i))*n], 0)
    self.tile: tf.constant = tf.constant(tile, self.dtype)
    """
