from typing import *
import networkx as nx
import numpy as np
import scipy as sp

SparseArray = sp.sparse.sparray

# ┏━┓╺┳┓ ┏┓┏━┓┏━╸┏━╸┏┓╻┏━╸╻ ╻   ┏┳┓┏━┓╺┳╸┏━┓╻┏━╸┏━╸┏━┓
# ┣━┫ ┃┃  ┃┣━┫┃  ┣╸ ┃┗┫┃  ┗┳┛   ┃┃┃┣━┫ ┃ ┣┳┛┃┃  ┣╸ ┗━┓
# ╹ ╹╺┻┛┗━┛╹ ╹┗━╸┗━╸╹ ╹┗━╸ ╹    ╹ ╹╹ ╹ ╹ ╹┗╸╹┗━╸┗━╸┗━┛
def apply_permutation(perm: tuple[int], matrix: np.ndarray) -> np.ndarray:
    return (matrix[perm, :])[:, perm]


def sparse_adj_to_lexicographic(mat: SparseArray) -> bool:
    """Returns wether mat_a > mat_b in terms of graph lexicographic order."""
    x, y = sp.sparse.triu(mat).coords  # coordonnées remplies dans mat
    lexico = np.empty((x.size + y.size,), dtype=x.dtype)
    lexico[0::2] = x
    lexico[1::2] = y
    return lexico

def np_lexicographic_gt(a: np.ndarray, b: np.ndarray) -> np.bool_:
    # make sure a and b have the same length :
    if a.size != b.size:
        if a.size < b.size:
            a = np.pad(a, (0, b.size - a.size))
        if b.size < a.size:
            b = np.pad(b, (0, a.size - b.size))
    # index of the first non-matching elements
    # idx = np.where(a != b)[0][0]
    idx = np.where((a > b) != (a < b))[0][0]  # version that handles NaN

    return a[idx] > b[idx]

def sparse_lexicographic_gt(mat_a: SparseArray, mat_b: SparseArray) -> np.bool_:
    return np_lexicographic_gt(sparse_adj_to_lexicographic(mat_a),
                               sparse_adj_to_lexicographic(mat_b))

if __name__ == '__main__':
    A = nx.Graph([(1, 2), (2, 3), (2, 4)])
    B = nx.Graph([(1, 3), (1, 4), (2, 3), (3, 4)])

    mA = nx.adjacency_matrix(A)
    mB = nx.adjacency_matrix(B)

    print(mA)
    print(mB)
    print(sparse_adj_to_lexicographic(mA))
    print(sparse_adj_to_lexicographic(mB))
    print(sparse_lexicographic_gt(mA, mB))


