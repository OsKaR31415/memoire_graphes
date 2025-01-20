from typing import *
import itertools as it
from functools import lru_cache
import math

import networkx as nx
import numpy as np

import matplotlib.pyplot as plt
from matplotlib import colormaps as cmaps

from time import time

def duration(function):
    def aux(*args, **kwargs):
        dep = time()
        result = function(*args, **kwargs)
        end = time()
        return result, end-dep
    return aux


def degrees_array(graph: nx.Graph) -> np.ndarray:
    """Return a numpy array containing the degrees of each node in the graph."""
    # return nx.adjacency_matrix(graph).sum(axis=0)  # faster ??
    return np.array(list(dict(graph.degree).values()))


def is_regular(graph: nx.Graph) -> bool | np.bool:
    """Returns True if the graph is regular i.e. all its degrees are equal."""
    if graph.number_of_edges() == 0 or graph.number_of_nodes() == 0:
        return True
    degrees = degrees_array(graph)
    return np.all(degrees == degrees[0])

def is_isomorphic(graph: nx.Graph, other_graph: nx.Graph) -> bool:
    if nx.faster_could_be_isomorphic(graph, other_graph):
        if nx.fast_could_be_isomorphic(graph, other_graph):
            if nx.is_isomorphic(graph, other_graph):
                return True
    return False


def is_duplicate(possible_duplicate: nx.Graph,
                 reference_graph_list: list[nx.Graph]) -> bool:
    """Check if graph is already in graph_list to the isomorphism, i.e. if
    graph is duplicate compared to graphs already in graph_list."""
    for graph in reference_graph_list:
        if is_isomorphic(possible_duplicate, graph):
            return True
    return False

def longest_cycle(graph: nx.Graph) -> list:
    return max(nx.simple_cycles(graph), key=len, default=[])


def sum_of_bits(number: int) -> int:
    total = 0
    while number > 0:
        total += number%2
        number >>= 1
    return total

# @lru_cache(maxsize=1024)
def graph_hash(graph: nx.Graph) -> int:
    return int(nx.weisfeiler_lehman_graph_hash(graph), 16)


def nodes_degree_classes(graph: nx.Graph) -> dict[int, list[int]]:
    """Return a dict mapping {degree -> nodes} for the given graph.
    This returns the equivalence classes for the relation "has the same degree".
    """
    degree_classes = dict()
    for node, degree in graph.degree:
        degree_classes[degree] = degree_classes.get(degree, []) + [node]
    return degree_classes


def neighbors_classes(graph: nx.Graph) -> dict[set[int], list[int]]:
    """Return a dict mapping {neighbors -> node} for the given graph.
    This returns the equivalence classes for the relation "has the same neighbors"."""
    neighbors_classes = dict()
    for node in graph.nodes:
        neigh = tuple(sorted(graph.neighbors(node)))  # set of neighbors of node
        neighbors_classes[neigh] = neighbors_classes.get(neigh, []) + [node]
    return neighbors_classes




# ┏━╸┏━┓┏━┓┏━┓╻ ╻   ╻ ╻╻┏━┓╻ ╻┏━┓╻  ╻┏━┓┏━┓╺┳╸╻┏━┓┏┓╻
# ┃╺┓┣┳┛┣━┫┣━┛┣━┫   ┃┏┛┃┗━┓┃ ┃┣━┫┃  ┃┗━┓┣━┫ ┃ ┃┃ ┃┃┗┫
# ┗━┛╹┗╸╹ ╹╹  ╹ ╹   ┗┛ ╹┗━┛┗━┛╹ ╹┗━╸╹┗━┛╹ ╹ ╹ ╹┗━┛╹ ╹

def graph_to_latex(graph: nx.Graph) -> str:
    """Return a string containg the LaTeX code that draws the given `graph`.
    This is a simple wrapper around nx.to_latex_raw, that has better default
    options, and better indentation."""
    # use networkx's function to export any nx.Graph to latex code
    latex: str = nx.to_latex_raw(
        graph,
        tikz_options="semithick",
        default_node_options="every node/.style={draw, circle}")

    latex_lines: list[str] = latex.split("\n")  # list of lines

    # re-indent properly the code, based on environments : \begin{...} ... \end{...}
    indent = 0  # the indentation counter
    for idx, line in enumerate(latex_lines):
        indent -= "\\end" in line  # decrease indent BEFORE the line containing \end{...}
        latex_lines[idx] = 4 * " " * indent + latex_lines[idx].lstrip()
        indent += "\\begin" in line  # increase indent AFTER the line containing \begin{...}

    return '\n'.join(latex_lines)

def plot_graphs(graph_list: Iterable[nx.Graph],
                savefig: bool =False,
                figpath: str =None) -> None:
    """Plot all given graphs using matplotlib."""
    # plt.ioff()
    N = len(graph_list)  # number of graphs to plot
    # number graphs per row / column. The sqrt makes the grid quite square
    rows = math.floor(math.sqrt(N))
    cols = math.ceil(N / rows)

    # initialize axes
    fig, axes = plt.subplots(rows, cols,
                             constrained_layout=False)

    # empty all axes and set bigger margins
    for ax in axes.flat:
        ax.axis("off")
        ax.margins(.2)

    # colors of the graphs
    COLORS = it.cycle(cmaps['Dark2'].colors)


    for ax, regraph, color in zip(axes.flat, graph_list, COLORS):

        # pos = nx.arf_layout(regraph,
        #                     scaling=3,
        #                     a=1.001,  # strength of springs between nodes
        #                     )
        # pos = nx.spring_layout(regraph)
        pos = nx.circular_layout(regraph)
        nx.draw(regraph,
                ax=ax,
                pos=pos,
                node_size=20,
                node_color=[color],
                edge_color=[color],
                width=1,
                )

    if savefig:
        plt.savefig(figpath)

    plt.show(block=True)

