# first line: 54
@memory.cache
def graph_hash(graph: nx.Graph) -> int:
    return int(nx.weisfeiler_lehman_graph_hash(graph), 16)
