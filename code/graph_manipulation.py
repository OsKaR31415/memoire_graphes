# from typing import 
import networkx as nx
import numpy as np
from numpy.typing import NDArray
import scipy as sp

SparseArray = sp.sparse.sparray

# ┏━┓╺┳┓ ┏┓┏━┓┏━╸┏━╸┏┓╻┏━╸╻ ╻   ┏┳┓┏━┓╺┳╸┏━┓╻┏━╸┏━╸┏━┓
# ┣━┫ ┃┃  ┃┣━┫┃  ┣╸ ┃┗┫┃  ┗┳┛   ┃┃┃┣━┫ ┃ ┣┳┛┃┃  ┣╸ ┗━┓
# ╹ ╹╺┻┛┗━┛╹ ╹┗━╸┗━╸╹ ╹┗━╸ ╹    ╹ ╹╹ ╹ ╹ ╹┗╸╹┗━╸┗━╸┗━┛
def apply_permutation(perm: tuple[int], matrix: np.ndarray) -> np.ndarray:
    return (matrix[perm, :])[:, perm]

def sparse_adj_to_canonical(mat: SparseArray) -> np.ndarray:
    """Convert a sparse adjacency matrix into a canonical representation."""
    x, y = sp.sparse.triu(mat).coords  # coordonnées remplies dans mat
    lexico = np.empty((x.size + y.size,), dtype=x.dtype)
    lexico[0::2] = x
    lexico[1::2] = y
    return lexico

def np_lexicographic_gt(a: NDArray[np.int32], b: NDArray[np.int32]) -> np.bool_:
    """Test if a > b in terms of lexicographic order."""
    # make sure a and b have the same length :
    # NOTE: There is probably a better way to do this, but we usually only need
    # to compare arrays with the same length
    if a.size != b.size:
        if a.size < b.size:
            a = np.pad(a, (0, b.size - a.size))
        if b.size < a.size:
            b = np.pad(b, (0, a.size - b.size))
    # index of the first non-matching elements
    # idx = np.where(a != b)[0][0]
    idx = np.where((a > b) != (a < b))[0][0]  # version that handles NaN
    # compare arrays at their first different value
    return a[idx] > b[idx]

def sparse_lexicographic_gt(mat_a: SparseArray, mat_b: SparseArray) -> np.bool_:
    """Return whether mat_a > mat_b regarding the lexicographic order."""
    return np_lexicographic_gt(sparse_adj_to_canonical(mat_a),
                               sparse_adj_to_canonical(mat_b))

def sparse_lexicographic_eq(mat_a: SparseArray, mat_b: SparseArray) -> np.bool_:
    """Return wether mat_a = mat_b regarding the lexicographic order."""
    return np.all(
        sparse_adj_to_canonical(mat_a) == sparse_adj_to_canonical(mat_b)
    )



if __name__ == '__main__':
    A = nx.Graph([(1, 2), (2, 3), (2, 4)])
    B = nx.Graph([(1, 3), (1, 4), (2, 3), (3, 4)])

    mA = nx.adjacency_matrix(A)
    mB = nx.adjacency_matrix(B)

    print(mA)
    print(mB)
    print(sparse_adj_to_canonical(mA))
    print(sparse_adj_to_canonical(mB))
    print(sparse_lexicographic_gt(mA, mB))




