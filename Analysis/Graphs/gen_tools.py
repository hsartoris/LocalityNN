import numpy as np

"""Graph property measures simple enough to group together."""

def order(mat: np.ndarray) -> int:
    """Return order of graph, where order is number of vertices."""
    return mat.shape[0]

def size(mat: np.ndarray) -> int:
    """Return size of graph, where size is number of edges."""
    return np.count_nonzero(mat)


