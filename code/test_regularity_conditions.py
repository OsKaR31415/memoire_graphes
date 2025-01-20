import networkx as nx
from time import time


def self_or_complement_contains_biggest_cycle(graph: nx.Graph) -> bool:
    self_cycles = nx.simple_cycles(graph)
    self_biggest_cycle = len(max(self_cycles, key=len, default=[]))
    complement_cycles = nx.simple_cycles(nx.complement(graph))
    complement_biggest_cycle = len(max(complement_cycles, key=len, default=[]))

    nb_nodes = len(graph.nodes)
    if self_biggest_cycle < nb_nodes and complement_biggest_cycle < nb_nodes:
        return False
    return True


def main():
    TOTAL_NB_GRAPHS = len(nx.graph_atlas_g())
    count = 0
    total_time = 0
    for graph in nx.graph_atlas_g():
        dep = time()
        test = self_or_complement_contains_biggest_cycle(graph)
        end = time()
        total_time += end - dep
        if not test:
            count += 1
    # print(f"{count} / {TOTAL_NB_GRAPHS} = {count/TOTAL_NB_GRAPHS}")
    print(f"took {total_time}s, avg {total_time / TOTAL_NB_GRAPHS}")


if __name__ == '__main__':
    for _ in range(10):
        main()
