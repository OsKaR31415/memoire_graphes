# first line: 67
@memory.cache
def parallel_search_regular_graphs(graph_set: GraphSet,
                                   max_degree: int,
                                   n_jobs: int =8,
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
                # TODO: find if there is a way to test the predicate that doesn't slow down the search too much
                # if not dead_end_predicate(new_graph):
                collecting_set.insert(new_graph)

    new_graphs = GraphSet()  # shared memory set of newly found graphs

    Parallel(n_jobs=n_jobs, require="sharedmem")(
        delayed(collect_graphs_with_one_more_edge_than)(graph, max_degree, new_graphs)
        for graph in graph_set.values()
    )

    # new_graphs.remove_where(dead_end_predicate)

    end = time()

    if verbose: print(f"found {len(new_graphs)} new graphs in {end - dep}s")

    return new_graphs
