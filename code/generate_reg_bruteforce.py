from typing import Callable
import networkx as nx
import numpy as np
import itertools as it

from graph_set import GraphSet


from utils import is_isomorphic, is_regular, plot_graphs, longest_cycle

from tqdm import tqdm
from time import time

from joblib import Parallel, delayed, Memory
memory = Memory("cache", verbose=0)


# def respects_size_limit(graph: nx.Graph) -> bool:
#     """Returns whether size <= max_degree * nb_nodes / 2
#     A d-regular graph with n nodes is of size dn/2.
#     Therefore, you cannot get a regular graph from adding nodes to a graph
#     where this predicate is false.
#     """
#     return graph.size() * 2 <= np.max(degrees_array(graph)) * graph.number_of_nodes()

def graph_contains_biggest_cycle(graph: nx.Graph) -> bool:
    """Returns whether the graph or its complements contains a n-cycle where n
    is the number of nodes in the graph."""
    nb_nodes = len(graph.nodes)
    if len(longest_cycle(graph)) == nb_nodes:
        return True
    if len(longest_cycle(nx.complement(graph))) == nb_nodes:
        return True
    return False


# def dead_end_predicate(graph: nx.Graph) -> bool:
#     """Predicate that returns True if the graph is a "dead end", i.e. if there
#     is no way to make it into a regular graph by adding edges.
#     The test has false negatives : it doesn't filter out every dead ends, but
#     it has no false positives : it keeps every graph that could become a
#     regular graph."""

def no_dead_end_predicate(graph: nx.Graph) -> bool:
    return True
    return (graph_contains_biggest_cycle(graph)
            or graph_contains_biggest_cycle(nx.complement(graph)))


# def custom_memoization(function):
#     hash_to_graph = dict()
#     memory = dict()
#     def memoized_function(graph: nx.Graph, *args) -> list[nx.Graph]:
#         graph_hash = nx.weisfeiler_lehman_graph_hash(graph)
#         if graph_hash not in memory:
#             memory[graph_hash] = function(graph, *args)
#             hash_to_graph[graph_hash] = graph
#             return memory[graph_hash]
#         if is_isomorphic(graph, hash_to_graph[graph_hash]):
#             # return memory[graph_hash]
#             return []
#         else:
#             return function(graph, *args)
#     return memoized_function


def search_regular_graphs(graph_set: GraphSet,
                          max_degree: int,
                          verbose: bool =True) -> GraphSet:
    """Returns the list of every possible graphs obtained from the graphs in
    *graphs* by adding one edge, to the isomophism, but limiting the degree of
    every node to the given max_degree.
    """

    dep = time()

    new_graphs = GraphSet(acception_predicate=no_dead_end_predicate)  # shared memory set of newly found graphs

    for graph in graph_set.values():
        free_dep_nodes = {node for node, degree in graph.degree
                          if degree < max_degree}
        for dep_node in free_dep_nodes:  # take every possible node in the graph
            possible_end_nodes = free_dep_nodes - {dep_node} - set(graph[dep_node])
            for end_node in possible_end_nodes:
                new_graph = graph.copy()
                new_graph.add_edge(dep_node, end_node)
                new_graphs.insert(new_graph)

    end = time()

    if verbose: print(f"found {len(new_graphs)} new graphs in {end - dep}s")

    return new_graphs


# @memory.cache
def parallel_search_regular_graphs(parallel: Parallel,
                                   graph_set: GraphSet,
                                   max_degree: int,
                                   verbose: bool =True) -> GraphSet:
    """Returns the list of every possible graphs obtained from the graphs in
    *graphs* by adding one edge, to the isomophism, but limiting the degree of
    every node to the given max_degree.
    """

    dep = time()

    def collect_graphs_with_one_more_edge_than(original_graph: nx.Graph,
                                               max_degree: int,
                                               collecting_set: GraphSet):
        free_dep_nodes = {node for node, degree in original_graph.degree
                          if degree < max_degree}
        for dep_node in free_dep_nodes:  # take every possible node in the graph
            possible_end_nodes = free_dep_nodes - {dep_node} - set(original_graph[dep_node])
            for end_node in possible_end_nodes:
                new_graph = original_graph.copy()
                new_graph.add_edge(dep_node, end_node)
                collecting_set.insert(new_graph)

    new_graphs = GraphSet(acception_predicate=no_dead_end_predicate)  # shared memory set of newly found graphs

    collector = delayed(collect_graphs_with_one_more_edge_than)
    parallel(
        collector(graph, max_degree, new_graphs)
        for graph in graph_set.values()
    )

    # for graph in graph_set.values():
    #     collect_graphs_with_one_more_edge_than(graph, max_degree, new_graphs)

    end = time()

    if verbose: print(f"found {len(new_graphs)} new graphs in {end - dep}s")

    return new_graphs


# @memory.cache
def list_regular_graphs(nb_nodes: int,
                        verbose: bool =True) -> list[nx.Graph]:
    dep_time = time()

    G = nx.Graph()
    G.add_nodes_from(list(range(nb_nodes)))

    MAX_DEGREE = (nb_nodes - 1) // 2

    # the max size that we need to search for
    SIZE = int(nb_nodes * MAX_DEGREE / 2)

    if verbose:
        print(f"Generating regular graphs with {nb_nodes} nodes")
        print(f"maximum size = {SIZE}, maximum degree = {MAX_DEGREE}")

    graphs_of_size_i = GraphSet([G], acception_predicate=no_dead_end_predicate)
    found = graphs_of_size_i.copy()
    found.acception_predicate = is_regular  # accept only regular graphs

    # # PARALLEL VERSION{{{
    # with Parallel(n_jobs=8, require="sharedmem") as parallel:
    #     for i in range(SIZE):
    #         if verbose: print(f"size = {i}/{SIZE}")
    #         graphs_of_size_i = parallel_search_regular_graphs(
    #             parallel,
    #             graphs_of_size_i,
    #             max_degree=MAX_DEGREE,
    #             verbose=verbose,
    #         )
    #         found.update(graphs_of_size_i)}}}

    # NON-PARALLEL VERSION
    total_update_time = 0
    max_nb_graphs_with_same_size = 0
    for i in range(SIZE):
        if verbose: print(f"size = {i}/{SIZE}")
        graphs_of_size_i = search_regular_graphs(
            graphs_of_size_i,
            max_degree=MAX_DEGREE,
            verbose=verbose,
        )
        length = len(graphs_of_size_i)
        if length > max_nb_graphs_with_same_size:
            max_nb_graphs_with_same_size = length
        update_dep = time()
        found.update(graphs_of_size_i)
        update_end = time()
        total_update_time += update_end - update_dep

    if verbose:
        print(f"accumulation of graphs took {total_update_time}s")
        print(f"there were {max_nb_graphs_with_same_size} graphs in the biggest iteration")

    # no need to filter since the graphset already has an acception predicate !
    # found = GraphSet(filter(is_regular, found.values()))

    complements = GraphSet(nx.complement(graph) for graph in found.values())
    found += complements

    found = list(sorted(found.values(), key=lambda g: g.degree[0]))
    if verbose: print("found", len(found), "regular graphs")

    end_time = time()
    if verbose: print(f"this took {end_time - dep_time}s")

    return found


if __name__ == '__main__':
    NB_NODES = 5

    regular_graphs = list_regular_graphs(NB_NODES, verbose=True)
    plot_graphs(regular_graphs,
                figpath=f"img/{NB_NODES}_nodes_regular_graphs.png",
                savefig=False)

    # for n in range(1, 9):
    #     regular_graphs = list_regular_graphs(n, verbose=False)
    #     print(f"n = {n}   R(n) = {len(regular_graphs)}")
    #     # plot_graphs(found)
