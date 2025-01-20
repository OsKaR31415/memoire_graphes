import networkx as nx
import numpy as np
import itertools as it

from utils import is_regular, plot_graphs

from string import ascii_uppercase


# magic variables (aliases) for graphs nodes
alphabet = list(ascii_uppercase)
A, B, C, D, E, F, G, H, I, J, K, L, M, N, O, P, Q, R, S, T, U, V, W, X, Y, Z = alphabet

NODES = (A, B, C, D, E, F)
EDGES = tuple(it.combinations(NODES, 2))

# Adj = [[0, 1, 1, 0],
#        [1, 0, 0, 1],
#        [1, 0, 0, 1],
#        [0, 1, 1, 0]]
# Adj = np.array(Adj, dtype=bool)
# Gr = nx.from_numpy_array(Adj, create_using=nx.Graph)

graphs = nx.graph_atlas_g()

print(len(graphs), "graphs loaded.")


regular_graphs = filter(is_regular, graphs)
# regular_graphs = sorted(regular_graphs, key=lambda graph: (graph.number_of_nodes(), graph.number_of_edges())[::-1])
regular_graphs = list(regular_graphs)

print("found", len(regular_graphs), "regular graphs")

plot_graphs(regular_graphs)


