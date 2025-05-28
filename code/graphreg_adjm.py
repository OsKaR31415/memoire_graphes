from networkx import normalized_cut_size
import numpy as np
from sys import setrecursionlimit
from utils import is_regular, print_madj
from canonicity import canonical_repr, is_canonical, naive_find_canonical
import graph_manipulation as gm



def f_zero(k: int, t: int):
    I = np.arange(0, t//2)
    if t % 2 == 1:
        return 1 + k * np.sum((k-1) ** I)
    return 2 * np.sum((k-1) ** I)

def ordrek(graph: np.ndarray,
           degrees: list[int],
           last_inserted_edge_x: int,
           last_inserted_edge_y: int,
           first_isolated_node: int,
           depth: int =0):
    indent = "->" * depth
    print_madj(graph)
    # print(f"{indent}{canonical_repr(graph)}")
    print(f"{indent}{degrees}")
    print(f"{indent}{gm.degrees_of(graph)}")
    # print(f"{indent}{last_inserted_edge_x}, {last_inserted_edge_y}   {first_isolated_node}")

    # test if the graph can be extended to a regular graph
    # criteria in Lemma 3.2.3.
    last_inserted_edge_start_degree = degrees[last_inserted_edge_x]
    if (last_inserted_edge_y+1 > n - k
            and degrees[last_inserted_edge_x] < k
            and n - last_inserted_edge_y+1 < k - degrees[last_inserted_edge_x]):
        return [] # No regular graphs can be created by adding edges

    # criteria in Lemma 3.2.4.
    if (last_inserted_edge_x+1 >= n - k
            and last_inserted_edge_start_degree == k):
        for i in range(last_inserted_edge_y, n): # ]y, n]
            if n - last_inserted_edge_x+1 - 1 < k - degrees[i]:
                return [] # No regular graphs can be created by adding edges

    new_edge_x = last_inserted_edge_x
    new_edge_y = last_inserted_edge_y

    # find the next node with degree != k
    while new_edge_x < n-1 and degrees[new_edge_x] == k:
        new_edge_x += 1

    # update the first_isolated_node
    if first_isolated_node <= new_edge_y:
        first_isolated_node = new_edge_y + 1

    # # eliminate non-connected candidates
    # if new_edge_x == first_isolated_node:
    #     return []

    # input(',')
    # if not is_canonical(graph):
    #     print('.')
    #     return []

    # # return if graph is regular
    # if new_edge_x+1 == n and degrees[new_edge_x] == k:
    #     if not gm.is_regular(graph):
    #         print_madj(graph)
    #         print(gm.degrees_of(graph))
    #         input("problem")
    #     print("is regular")
    #     # input()
    #     return [graph]
    # # input()

    if gm.is_regular(graph):
        print("is regular")
        return [graph]

    new_edge_y = new_edge_x

    result = []
    while new_edge_y+1 < n:
        new_edge_y += 1
        # print(indent, "█", new_edge_x, new_edge_y)
        if degrees[new_edge_y] < k and not graph[new_edge_x,new_edge_y]:
            gm.insert_edge(graph, new_edge_x, new_edge_y)
            degrees[new_edge_x] += 1
            degrees[new_edge_y] += 1
            # print(indent, end="")
            # print_madj(graph)
            # print(indent, *degrees, sep="")
            result.extend(ordrek(graph=graph.copy(),
                                 degrees=degrees.copy(),
                                 last_inserted_edge_x=new_edge_x,
                                 last_inserted_edge_y=new_edge_y,
                                 first_isolated_node=first_isolated_node,
                                 depth=depth+1))
            gm.remove_edge(graph, new_edge_x, new_edge_y)
            degrees[new_edge_x] -= 1
            degrees[new_edge_y] -= 1
    print(indent, "≡", len(result))
    # print("see :")
    # print_madj(result[-1])
    return result



# def row_criterion_gen(i: int):
    



if __name__ == '__main__':
    # print(f_zero(3, 3))

    # setrecursionlimit(20000)
    n = 10
    k = 3

    G = np.zeros((n, n), dtype=bool)

    # # k = 2, t = 4
    # gm.insert_edge(G, 0, 1)
    # gm.insert_edge(G, 0, 2)
    # gm.insert_edge(G, 1, 3)

    # # k=3, t=4 ==> n=6
    # gm.insert_edge(G, 0, 1)
    # gm.insert_edge(G, 0, 2)
    # gm.insert_edge(G, 0, 3)
    # gm.insert_edge(G, 1, 4)
    # gm.insert_edge(G, 1, 5)

    # k=3, t=5 ==> n=10
    gm.insert_edge(G, 0, 1)
    gm.insert_edge(G, 0, 2)
    gm.insert_edge(G, 0, 3)
    gm.insert_edge(G, 1, 4)
    gm.insert_edge(G, 1, 5)
    gm.insert_edge(G, 2, 6)
    gm.insert_edge(G, 2, 7)
    gm.insert_edge(G, 4, 8)
    gm.insert_edge(G, 4, 9)

    # # k=3, n=10
    # gm.insert_edge(G, 0, 1)
    # gm.insert_edge(G, 0, 2)
    # gm.insert_edge(G, 0, 3)
    # gm.insert_edge(G, 1, 4)
    # gm.insert_edge(G, 1, 5)
    # gm.insert_edge(G, 2, 6)
    # gm.insert_edge(G, 2, 7)
    # gm.insert_edge(G, 3, 8)
    # gm.insert_edge(G, 3, 9)

    # gm.insert_edge(G, 0, 1)
    print_madj(G)
    # print_madj(naive_find_canonical(G))

    # degrees = np.zeros(n)
    # degrees[0] = k
    # degrees[1:k] = 1  # [1:k[ = [1:k+1]
    # degrees[k+1:n] = 0

    degrees = gm.degrees_of(G)

    O = ordrek(G, degrees, 0, 1, 2)
    print("-" * 50)
    for g in O:
        print_madj(g)
        print(gm.is_regular(g))
        # print(degrees_of(g))
    print(len(O), "graphs found")
