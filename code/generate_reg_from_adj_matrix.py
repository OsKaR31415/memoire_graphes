import networkx as nx
import numpy as np
from functools import cache
import itertools as it
import multiprocessing as mp

from utils import duration


def print_madj(madj: np.ndarray) -> None:
    """Pretty printing of an adjacency matrix."""
    # res = "┏" + "━" * len(madj) + "┓\n"
    res = "┏" + "".join(str(n%10) for n in range(len(madj))) + "┓\n"
    for ln_idx, line in enumerate(madj):
        res += str(ln_idx % 10)
        for col_idx, elt in enumerate(line):
            res += "█" if elt else "╲" if ln_idx == col_idx else "┼"
        res += "┃\n"
    res += "┗" + "━" * len(madj)
    res += "R\n" if is_regular(madj) else "┛\n"
    print(res)


def add_edge(madj: np.ndarray, dep: int, end: int) -> None:
    madj[dep, end] = np.True_
    madj[end, dep] = np.True_


def empty_graph(nb_nodes: int) -> np.ndarray:
    return np.zeros((nb_nodes, nb_nodes), np.bool)

def cyclic_graph(nb_nodes: int) -> np.ndarray:
    madj = empty_graph(nb_nodes)
    for i in range(nb_nodes):
        add_edge(madj, i, (i+1) % nb_nodes)
    return madj

def is_regular(madj: np.ndarray) -> np.bool:
    degrees = madj.sum(axis=0)
    return np.all(degrees == degrees[0])

@cache
def mask_regularity_indexes(nb_nodes) -> list[list[int]]:
    result = [[] for _ in range(nb_nodes)]
    counter = 0
    for offset in range(1, nb_nodes):
        for idx in range(offset, nb_nodes):
            result[idx].append(counter)
            counter += 1
    slice_start = 0
    for idx in range(0, nb_nodes):
        length = nb_nodes - idx - 1
        result[idx].extend(range(slice_start, slice_start + length))
        slice_start += length
    return result

def mask_is_regular(mask, nb_nodes: int) -> np.bool:
    mask = np.unpackbits(np.flip(np.array([mask]).view(np.uint8)))
    mask = mask[:-1-nb_nodes*(nb_nodes+1)//2:-1]
    indexes = mask_regularity_indexes(nb_nodes)
    degrees = mask[indexes].sum(axis=1)
    return np.all(degrees == degrees[0])


def graph_from_mask(mask: int, nb_nodes) -> np.ndarray:
    graph = empty_graph(nb_nodes)
    for dep in range(0, nb_nodes):
        for end in range(dep+1, nb_nodes):
            if mask % 2:
                add_edge(graph, dep, end)
            mask >>= 1
    return graph


def construct_regular_graphs(nb_nodes: int):
    for mask in range(2 ** (nb_nodes * (nb_nodes - 1) // 2)):
        if mask_is_regular(mask, nb_nodes):
            graph = graph_from_mask(mask, nb_nodes)
            print(mask)
            print_madj(graph)

def harverster(mask, nb_nodes):
    if mask_is_regular(mask, nb_nodes):
        return mask
    return None

def parallel_construct_regular_graphs(nb_nodes: int):


    with mp.Pool(processes=8) as pool:
        masks = range(2 ** (nb_nodes * (nb_nodes - 1) // 2))
        res = pool.starmap_async(harverster, zip(masks, it.repeat(nb_nodes)))
        # res = pool.apply_async(harverster, zip(masks, it.repeat(nb_nodes)))
        pool.close()
        pool.join()
    return [value for value in res.get() if value is not None]



if __name__ == '__main__':
    # print(duration(construct_regular_graphs)(7))
    print(duration(parallel_construct_regular_graphs)(7))

    # #n, mask = 6, 656
    # #n, mask = 6, 32111
    # n, mask = 7, 1106181
    # print_madj(graph_from_mask(mask, n))
    # print(mask_is_regular(mask, n))

