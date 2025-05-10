import networkx as nx
import numpy as np
import itertools as it

from graph_manipulation import sparse_lexicographic_gt

import scipy as sp

SparseArray = sp.sparse.sparray

A = nx.Graph([(1, 2), (2, 3), (2, 4)])
B = nx.Graph([(1, 3), (1, 4), (2, 3), (3, 4)])

mA = nx.adjacency_matrix(A)
mB = nx.adjacency_matrix(B)

print(mA)
print(mB)
print(sparse_lexicographic_gt(mA, mB))



