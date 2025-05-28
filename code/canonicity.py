import numpy as np
from itertools import permutations
import graph_manipulation as gm
from utils import permutation_product, print_madj

def canonical_repr(graph: np.ndarray) -> np.ndarray:
    """Return the canonical representation of graph."""
    where = np.nonzero(graph)  # find coordinates
    dep, end = where  # separate coordinates
    return np.array(where)[:,dep < end].transpose()  # filter only where dep < end

# def lexico_gt(a: np.ndarray, b: np.ndarray) -> np.bool_:
#     """Test if a > b in terms of lexicographic order.
#     This assumes that a and b have the same size."""
#     # make sure a and b have the same length :
#     # NOTE: There is probably a better way to do this, but we usually only need
#     # to compare arrays with the same length
#     if a.size != b.size:
#         if a.size < b.size:
#             a = np.pad(a, (0, b.size - a.size))
#         if b.size < a.size:
#             b = np.pad(b, (0, a.size - b.size))
#     # False when a equals b
#     if np.array_equal(a, b):
#         return np.False_
#     # index of the first non-matching elements
#     # idx = np.nonzero(a != b)[0][0]
#     idx = np.where((a > b) != (a < b))[0][0]  # version that handles NaN
#     # compare arrays at their first different value
#     return a[idx] > b[idx]

def lexico_gt(a: np.ndarray[np.int_], b: np.ndarray[np.int_]) -> np.bool_:
    return list(a) > list(b)

def lexico_geq(a, b):
    # if np.array_equal(a, b): return True
    # idx = np.nonzero(a != b)[0][0]
    # return a[idx] > b[idx]
    return list(a) >= list(b)

def madj_gt(a: np.ndarray, b: np.ndarray) -> np.bool_:
    """Adjacency matrix comparison : a > b"""
    return lexico_gt(canonical_repr(a).ravel(), canonical_repr(b).ravel())



def is_canonical(graph: np.ndarray) -> bool:
    return naive_canonicity_test(graph)

def naive_canonicity_test(graph: np.ndarray) -> bool:
    k = 0
    n = graph.shape[0]
    for permutation in permutations(range(n)):
        k += 1
        if not k % 10000:
            print(k)
        perm_graph = gm.apply_permutation(permutation, graph)
        if madj_gt(graph, perm_graph):
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
    result[i][i:] = Gamma[i][i:]
    # for end_node in range(n):
    #     result[i,end_node] = Gamma[i,end_node]
    return result

def naive_C_i(Gamma_i: np.ndarray, i: int, n: int):
    """
    /!\\ permutations indexées à 0
    """
    print(">", i)
    assert i >= 1
    if i == 1:
        # renvoyer C_S_n({0})
        # les permutations de 𝔖_n qui laissent 0 invariant
        return {(0,) + pi
                for pi in permutations(range(1, n))}
    # renvoyer C_N_i({1, ..., i+1})
    return {pi for pi in naive_N_i(Gamma_i, i-1, n)
            if pi[i] == i}

def naive_N_i(Gamma_i: np.ndarray, i: int, n: int):
    return {perm for perm in naive_C_i(Gamma_i, i, n)
            if np.all(gm.apply_permutation(perm, Gamma_i) == Gamma_i)}


def canonical_refinement(x: np.ndarray, mu: np.ndarray):
    """Return the canonical refinment of x with respect to mu.
    Args:
        x (1D boolean array): a boolean vector (usually of size n)
        mu (1D integer array): a partition of x.size
    """
    mu_cumsum = np.concat(([0], np.cumsum(mu)))
    # the list of mu bock slicing indexes.
    # has to be a tuple (not a zip generator) because it is used multiple times
    mu_blocks = tuple(zip(mu_cumsum[:-1], mu_cumsum[1:]))
    canonizing_perm = np.arange(x.size)
    for dep, end in mu_blocks:
        perm = dep + np.argsort(np.invert(x[dep:end]))
        canonizing_perm[perm] = np.arange(dep, end)
    canonical_x = x[canonizing_perm]
    # the actual canonical refinment of x with respect to mu
    refinement = []
    for dep, end in mu_blocks:
        block = canonical_x[dep:end]
        zeroes = np.count_nonzero(0 == block)
        ones = block.size - zeroes
        if ones > 0:
            refinement.append(ones)
        if zeroes > 0:
            refinement.append(zeroes)
    return refinement


def naive_is_semicanonical(Gamma: np.ndarray) -> bool:
    for i in range(0, n):
        Gi = Gamma_i(Gamma, i)
        for perm in naive_C_i(Gi, i, n):
            perm_Gi = gm.apply_permutation(perm, Gi)
            if not gm.np_lexicographic_gt(perm_Gi.ravel(), Gi.ravel()):
                return False
    return True



def katest(adjm: np.ndarray, kmn: np.ndarray, tz):
    """
    Args:
        adjm (2D np array): The adjacency matrix of the studied graph.
        kmn (permutation of S_n): should be initialized to the identity.
        tz (int): ?? recursion depth ??
    """
    found_automorphisms = []  # modified in place /!\ side effect
    for i in reversed(range(1, n+1)): # n to 1
        for j in range(i+1, n+1): # i+1 to n
            kmn[((i, j),)] = kmn[((j, i),)]
            # kmn[i] = j
            # kmn[j] = i
            erg = karek(i+1, adjm, kmn, found_automorphisms)
            kmn[((i, j),)] = kmn[((j, i),)]
            # kmn[i] = i
            # kmn[j] = j
            if erg == -1:
                return False
    return True

def karek(tz: int, adjm, kmn, found_automorphisms)
    if tz == n:
        adjvgl(adjm, kmn, found_automorphisms)

    for j in J(tz):
        alpha = alpha_of(tz, j)
        if np.all(permutation_product(alpha, kmn) == alpha):
            continue
        karek(tz + 1, adjm, kmn, found_automorphisms)
    return

def adjvgl(adjm, kmn, found_automorphisms):
    perm_adjm = gm.apply_permutation(kmn, adjm)
    if np.all(perm_adjm == adjm):
        # kmn is an automorphism
        found_automorphisms.append(kmn)
        return 0
    if gm.np_lexicographic_gt(perm_adjm.ravel(), adjm.ravel()):
        # the graph is not canonical
        return -1
    return 1



if __name__ == '__main__':
    n = 8

    G = np.zeros((n, n), dtype=bool)

    gm.insert_edge(G, 0, 1)
    gm.insert_edge(G, 0, 2)
    gm.insert_edge(G, 1, 3)

    # H = np.zeros((5, 5), dtype=bool)
    # gm.insert_edge(H, 0, 1)

    # canonical_G = naive_find_canonical(G)
    # print(canonical_repr(canonical_G))
    # print(naive_canonicity_test(G))
    # print(naive_canonicity_test(canonical_G))

    i = 0
    Gi = Gamma_i(G, i)
    print_madj(Gi)
    # print(naive_C_i(Gi, i, n))
    x = np.array((0, 1, 0, 0, 0), dtype=bool)
    mu = (1, 4)
    print(canonical_refinement(x, mu))
    print(naive_is_semicanonical(G))




