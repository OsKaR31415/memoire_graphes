import numpy as np
from itertools import permutations
import graph_manipulation as gm
from utils import print_madj

def canonical_repr(graph: np.ndarray) -> np.ndarray:
    """Return the canonical representation of graph."""
    where = np.array(np.nonzero(graph))  # find coordinates
    dep, end = where  # separate coordinates
    return where[:,dep < end].transpose()  # filter only where dep < end

def lexico_gt(a: np.ndarray, b: np.ndarray) -> np.bool_:
    """Test if a > b in terms of lexicographic order.
    This assumes that a and b have the same size."""
    # make sure a and b have the same length :
    # NOTE: There is probably a better way to do this, but we usually only need
    # to compare arrays with the same length
    if a.size != b.size:
        if a.size < b.size:
            a = np.pad(a, (0, b.size - a.size))
        if b.size < a.size:
            b = np.pad(b, (0, a.size - b.size))
    # False when a equals b
    if np.array_equal(a, b):
        return np.False_
    # index of the first non-matching elements
    # idx = np.nonzero(a != b)[0][0]
    idx = np.where((a > b) != (a < b))[0][0]  # version that handles NaN
    # compare arrays at their first different value
    return a[idx] > b[idx]

def madj_gt(a: np.ndarray, b: np.ndarray) -> np.bool_:
    """Adjacency matrix comparison : a > b"""
    return lexico_gt(canonical_repr(a).ravel(), canonical_repr(b).ravel())


def lexico_geq(a, b):
    if np.array_equal(a, b): return True
    idx = np.nonzero(a != b)[0][0]
    return a[idx] > b[idx]

def is_canonical(graph: np.ndarray) -> bool:
    return naive_canonicity_test(graph)

def naive_canonicity_test(graph: np.ndarray) -> bool:
    n = graph.shape[0]
    for permutation in permutations(range(n)):
        perm_graph = gm.apply_permutation(permutation, graph)
        if madj_gt(graph, perm_graph):
            print(canonical_repr(perm_graph))
            return False
    return True

def naive_find_canonical(graph: np.ndarray) -> np.ndarray:
    current_min = graph
    n = graph.shape[0]
    for permutation in permutations(range(n)):
        perm_graph = gm.apply_permutation(permutation, graph)
        if madj_gt(current_min, perm_graph):
            current_min = perm_graph
    return current_min

def Gamma_i(Gamma: np.ndarray, i: int) -> np.ndarray:
    """Retourne Γ_i pour Γ∈𝒢_n.
    Le graphe constitué des arrêtes de Gamma qui partent de i.
    Args:
        Gamma (madj): le graphe Γ
        i (int): l'indice
    """
    n = Gamma.shape[0]
    result = np.zeros_like(Gamma)
    for end_node in range(n):
        result[i,end_node] = result[end_node,i] = Gamma[i,end_node]
    return result

def N_i(Gamma_i: np.ndarray, i: int):
    return {perm for perm in C_i(i)
            if not lexico_gt(gm.apply_permutation(perm, Gamma_i), Gamma_i)}





if __name__ == '__main__':
    G = np.zeros((7, 7), dtype=bool)
    gm.insert_edge(G, 1, 2)
    gm.insert_edge(G, 1, 3)
    gm.insert_edge(G, 0, 2)
    gm.insert_edge(G, 3, 2)
    gm.insert_edge(G, 4, 1)

    H = np.zeros((5, 5), dtype=bool)
    gm.insert_edge(H, 0, 1)

    canonical_G = naive_find_canonical(G)
    print(canonical_repr(G))
    print(canonical_repr(canonical_G))
    print(naive_canonicity_test(G))
    print(naive_canonicity_test(canonical_G))




