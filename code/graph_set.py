from utils import graph_hash, is_duplicate, is_isomorphic, is_regular

import networkx as nx
from copy import deepcopy
from typing import Self, Iterable, Callable


def is_isomorphic(graph: nx.Graph, other_graph: nx.Graph) -> bool:
    if nx.faster_could_be_isomorphic(graph, other_graph):
        if nx.fast_could_be_isomorphic(graph, other_graph):
            if nx.is_isomorphic(graph, other_graph):
                return True
    return False


def graph_hash(graph: nx.Graph) -> int:
    return int(nx.weisfeiler_lehman_graph_hash(graph), 16)


class GraphSet(dict):
    def __init__(self, graphs: Iterable[nx.Graph] = [],
                 acception_predicate: Callable = None) -> None:
        """
        Args:
            graphs (Iterable[nx.Graph]): A list of graphs to put into the GraphSet.
            acception_predicate (Callable): The predicate that tells if a graph
                is accepted into the GraphSet. Defaults to accepting any graph.
        """
        if acception_predicate is None:
            self.acception_predicate = lambda _: True
        else:
            self.acception_predicate = acception_predicate
        for graph in graphs:
            self.insert(graph)

    def insert(self, graph: nx.Graph) -> None:
        if not self.acception_predicate(graph):
            return  # do not add the graph
        hashed_graph = graph_hash(graph)
        while self.get(hashed_graph) is not None:
            if is_isomorphic(self[hashed_graph], graph):
                return  # the graphs is already in the set
            hashed_graph += 1  # find nest available spot
        self[hashed_graph] = graph

    def insert_list(self, graphs: Iterable[nx.Graph]) -> None:
        for graph in graphs:
            self.insert(graph)

    def update(self, other: Self) -> None:
        for hashed_graph, graph in other.items():
            if not self.acception_predicate(graph):
                continue  # do not add the graph
            if hashed_graph in self:
                # The hash is already in self, so there might be an isomorphic
                # graph already in self.
                # You still have to test for that
                # this code is the same as in the self.insert method :
                while self.get(hashed_graph) is not None:
                    if is_isomorphic(self[hashed_graph], graph):
                        return  # the graphs is already in the set
                    hashed_graph += 1  # find nest available spot
                self[hashed_graph] = graph
            else:
                # The hash isn't in self, so there is no way an isomorphic
                # graph is already in self.
                # you can just insert the graph as is.
                self[hashed_graph] = graph

    def contains(self, graph: nx.Graph) -> bool:
        """Return whether item is in the set or not. The test works both on
        graphs and on graphs hashes (if you use the same hash).
        """
        hashed_graph = graph_hash(graph)
        while hashed_graph in self:
            if is_isomorphic(graph, self[hashed_graph]):
                return True
            hashed_graph += 1
        return False

    def __str__(self) -> str:
        res = "FoundGraphs{"
        for key, graph in self.items():
            res += f"<{str(graph)}>, "
        res = res[:-2]  # remove trailing ", "
        res += "}"
        return res

    def __repr__(self) -> str:
        res = "FoundGraphs("
        for key, graph in self.items():
            res += f"{repr(graph)}, "
        res = res[:-2]  # remove trailing ", "
        res += ")"
        return res

    def copy(self) -> Self:
        obj_copy = deepcopy(self)
        obj_copy.acception_predicate = self.acception_predicate
        return obj_copy

    def __add__(self, other: Self) -> Self:
        addition = self.copy()
        addition.update(other)
        return addition

    # def remove_where(self, predicate: Callable) -> None:
    #     keys_to_delete = set()
    #     for key, graph in self.items():
    #         if predicate(graph):
    #             keys_to_delete.add(key)
    #     for key in keys_to_delete:
    #         del self[key]

    def __getstate__(self) -> Self:
        """Used for serializing instances."""
        return self

    def __setstate__(self, state):
        """Used for de-serializing"""
        self.update(state)


def benchmark():
    from random import randint, choice
    from time import time

    rand_graph_args = {"n": 10,
                       "p": 0.1}

    N = 10_000
    graphs = GraphSet()
    total_time = 0
    for _ in range(N):
        dep = time()
        graphs.insert(nx.fast_gnp_random_graph(**rand_graph_args))
        end = time()
        total_time += end - dep
    print(f"insertion took {total_time/N} s/graph")

    graphs_tuple = tuple(graphs.values())

    NB_TESTS = 10_000
    total_time = 0
    for i in range(NB_TESTS):
        if i % 2:
            graph = nx.fast_gnp_random_graph(**rand_graph_args)
        else:
            graph = choice(graphs_tuple)
        dep = time()
        graphs.contains(graph)  # just do the test
        end = time()
        total_time += end - dep
    print(f"membership checking took {total_time / NB_TESTS} s/graph")

if __name__ == '__main__':
    predicate = lambda graph: len(graph.nodes) == 5
    predicate = is_regular

    regset = GraphSet([], predicate)
    atlaset = GraphSet(nx.graph_atlas_g())
    regset.update(atlaset)

    iregset = GraphSet([], predicate)
    iregset.insert_list(nx.graph_atlas_g())

    print(regset)
    print(len(regset), len(iregset))

