import networkx as nx
import numpy as np
from contextlib import contextmanager
from utils import graph_with_new_edge, is_regular, can_be_extented_to_regular, degrees_list


def P(graph: nx.Graph, nb_nodes: int|None =None):
    if nb_nodes is None:
        nb_nodes = max(graph.nodes)

    max_edge = max(graph.edges)
    # generating all graphs made from `graph` by adding an edge greater than max_edge
    # first edges with same start as max_edge
    for edge_end in range(max_edge[1]+1, nb_nodes+1):
        edge = (max_edge[0], edge_end)
        yield graph_with_new_edge(graph, edge)
    # then edges where edge_start > max_edge[0]
    for edge_start in range(max_edge[0]+1, nb_nodes+1):
        for edge_end in range(edge_start+1, nb_nodes+1):
            edge = (edge_start, edge_end)
            yield graph_with_new_edge(graph, edge)


def Q(graph: nx.Graph, nb_nodes: int|None =None):
    if nb_nodes is None:
        nb_nodes = max(graph.nodes)
    # iteratively apply P to all elements of current_P
    # yield all results (including)
    current_P = {graph,}
    yield graph
    # stop iterations when no more graphs can be generated
    new_graph_found = True
    while new_graph_found:
        new_graph_found = False
        new_P = set()
        for g in current_P:
            new_P.update(P(g))
            new_graph_found = True
        yield from new_P
        current_P=new_P

def ordrek(graph: nx.Graph,
           degrees: list[int],
           last_inserted_edge_x: int,
           last_inserted_edge_y: int,
           jirst_isolated_node: int):
    """
    Args:
        graph (nx.Graph):
        degrees (list[int]): The list containing the degree of each node in canonical order.
        last_inserted_edge_x (int): The last inserted edge' start node.
        last_inserted_edge_y (int): The last inserted edge's end node.
        first_isolated_node (int): The smallest node with degree 0 before the insertion of last_inserted_edge.

    """
    result = []  # list of found graphs

    # test if the graph can be extended to a regular graph
    # criteria in Lemma 3.2.3.
    last_inserted_edge_start_degree = degrees[last_inserted_edge_x]
    if (last_inserted_edge_y > n - k
            and degrees[last_inserted_edge_x] < k
            and n - last_inserted_edge_y < k - degrees[last_inserted_edge_x]):
        return [] # No regular graphs can be created by adding edges

    # # criteria in Lemma 3.2.4.
    if (last_inserted_edge_x >= n - k
            and last_inserted_edge_start_degree == k):
        for i in range(last_inserted_edge_y+1, n+1): # ]y, n]
            if n - last_inserted_edge_x - 1 < k - degrees[i]:
                return [] # No regular graphs can be created by adding edges

    new_edge_x = last_inserted_edge_x
    new_edge_y = last_inserted_edge_y

    # find the next node with degree != k
    while new_edge_x < n and degrees[new_edge_x] == k:
        new_edge_x += 1

    if first_isolated_node <= new_edge_y:
        first_isolated_node = new_edge_y + 1

    if new_edge_x == first_isolated_node:
        return [] # No regular graphs can be created by adding edges

    # TODO: implement canonicity test
    # if not is_canonical(graph): return

    if new_edge_x == n and degrees[new_edge_x] == k:
        print("=>", graph)
        result.append(graph)
        # yield graph

    # yield graph

    for new_edge_y in range(new_edge_x+1, n):
        print(new_edge_y)
        if degrees[new_edge_y] < k:
            graph[new_edge_x][new_edge_y] = 1
            graph[new_edge_y][new_edge_x] = 1
            degrees[new_edge_x] += 1
            degrees[new_edge_y] += 1
            recurse = ordrek(graph=graph,
                             degrees=degrees,
                             last_inserted_edge_x=new_edge_x,
                             last_inserted_edge_y=new_edge_y,
                             first_isolated_node=first_isolated_node)
            result.extend(recurse)
            graph[new_edge_x][new_edge_y] = 0
            graph[new_edge_y][new_edge_x] = 0
            degrees[new_edge_x] -= 1
            degrees[new_edge_y] -= 1
            # yield from recurse


    return result


@contextmanager
def add_edge_to_graph(graph: nx.Graph,
                      degrees: list[int],
                      edge_x: int, edge_y: int,
                      verbose: bool=False):
    if verbose: print(f"{graph.edges} + {edge_x}->{edge_y}")
    graph.add_edge(edge_x, edge_y)
    degrees[edge_x] += 1
    degrees[edge_y] += 1
    try:
        yield
    finally:
        if verbose: print(f"{graph.edges} - {edge_x}->{edge_y}")
        if not graph.has_edge(edge_x, edge_y):
            if verbose: print("the edge is not in the graph !")
            return
        graph.remove_edge(edge_x, edge_y)
        degrees[edge_x] -= 1
        degrees[edge_y] -= 1
        if verbose: (f"=> {graph.edges}")


if __name__ == '__main__':
    global n, k
    n = 6
    k = 3

    G = nx.Graph([(1, 2)])
    G.add_nodes_from(range(1, n+1))



    G = np.array([
        [0, 1, 0, 0, 0],
        [1, 0, 0, 0, 0],
        [0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0],
                  ])

    O = ordrek(G, G.sum(axis=0), 0, 1, 3)


    for g in O:
        if is_regular(g):
            print("-->", g)
        else:
            print(".", end="")

    print()



