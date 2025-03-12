import graph_tool.all as gt
from typing import List, Tuple
from graph_tool.inference import CliqueState
from general.graph_construction import convert_nx_to_gt

def get_cliques_from_subgraph(graph: gt.Graph) -> List[Tuple]:
    """
    Extracts hyperedges from a graph-tool subgraph using CliqueState.

    Parameters
    ----------
    graph : gt.Graph
        graph-tool Graph object representing the pathway or cluster of interest.

    Returns
    -------
    hyperedges : List[Tuple]
        List of tuples. Each tuple represents a hyperedges and stores the global node
        indices of the nodes it contains.
    """
    if graph.num_edges() == 0:
        return []

    state = CliqueState(graph)
    state.mcmc_sweep(niter=10000, beta=1)  # Burn-in
    min_entropy = float("inf")
    best_state = None

    for _ in range(1000):  # Run MCMC to find optimal clique state
        state.mcmc_sweep(niter=1, beta=1)  # Default inverse temperature
        current_entropy = state.entropy()
        if current_entropy < min_entropy:
            min_entropy = current_entropy
            best_state = state.copy()

    hyperedges = []
    for v in best_state.f.vertices():   # Iterate through factor graph
        if best_state.is_fac[v]:
            continue                    # Skip over factors (pairwise connections)
        if best_state.x[v] > 0:
            hyperedges.append(tuple(best_state.c[v]))

    return hyperedges


def extract_hyperedges(clusters, prefix=""):
    """
    Extracts hyperedges from each cluster after spectral clustering.

    Parameters
    ----------
    clusters : dict
        A hierarchical dictionary of clusters, where each entry is either a NetworkX graph or another dictionary of subclusters.
    prefix : str, optional
        Prefix for labeling clusters in a hierarchical structure.

    Returns
    -------
    results : dict
        A dictionary mapping cluster labels to hyperedges.
    """
    results = {}

    for label, subgraph in clusters.items():
        cluster_id = f"{prefix}{label}"

        if isinstance(subgraph, dict):
            # If it's a dictionary, recurse deeper
            results.update(extract_hyperedges(subgraph, prefix=f"{cluster_id}_"))
        else:
            # Convert to Graph-Tool and extract hyperedges
            gt_subgraph = convert_nx_to_gt(subgraph)
            results[cluster_id] = get_cliques_from_subgraph(gt_subgraph)

    return results
