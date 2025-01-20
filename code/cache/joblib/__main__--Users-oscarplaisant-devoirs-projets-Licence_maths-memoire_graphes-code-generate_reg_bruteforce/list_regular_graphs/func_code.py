# first line: 109
@memory.cache
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

    graphs_of_size_i = GraphSet([G])
    found = graphs_of_size_i.copy()
    for i in range(SIZE):
        if verbose: print(f"size = {i}/{SIZE}")
        graphs_of_size_i = parallel_search_regular_graphs(
            graphs_of_size_i,
            max_degree=MAX_DEGREE,
            n_jobs=2,
            verbose=verbose,
        )
        found.update(graphs_of_size_i)


    found = GraphSet(filter(is_regular, found.values()))

    complements = GraphSet(nx.complement(graph) for graph in found.values())
    found += complements

    found = list(sorted(found.values(), key=lambda g: g.degree[0]))
    if verbose: print("found", len(found), "regular graphs")

    end_time = time()
    if verbose: print(f"this took {end_time - dep_time}s")

    return found
