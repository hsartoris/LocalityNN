from typing import List
import numpy as np

"""Defines methods and helpers pertaining to breaking down a graph into isolated 
subgraphs, or components.

Not extensively tested.
"""

def components(mat: np.ndarray, threshold: float = 0) -> List[List[int]]:
    """Returns list of components within a graph.

    Components are groups of vertices connected by an edge of strength greater 
    than threshold.

    Components are returned as lists of ints, where each int is a vertex index 
    in the graph matrix.
    """

    # square matrix
    assert(mat.shape[0] == mat.shape[1])

    processed: List[int] = []

    components: List[List[int]] = []

    for i in range(mat.shape[0]):
        # skip if already in a component
        if i in processed: continue

        component: List[int] = []
        _component_recurse(i, mat, threshold, processed, component)
        components.append(component)
    
    return components
        

def _component_recurse(idx: int, mat: np.ndarray, threshold: float,
        processed: List[int], component: List[int]) -> None:
    """Performs depth-first search to complete component."""

    component.append(idx)
    processed.append(idx)

    for i in range(mat.shape[0]):
        if i in processed: continue

        if mat[idx, i] > threshold or mat[i, idx] > threshold:
            # connection of some direction between vertices i & idx; i has not 
            # been processed yet
            _component_recurse(i, mat, threshold, processed, component)
