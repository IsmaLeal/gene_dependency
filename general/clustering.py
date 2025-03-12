import numpy as np
import scipy.sparse
import networkx as nx
from sklearn.cluster import SpectralClustering

def optimal_spectral_clustering(A: np.ndarray, max_clusters: int = 20) -> int:
    """
    Determines the optimal number of clusters using the largest eigengap.

    Parameters
    ----------
    A : np.ndarray
        Adjacency matrix of the graph of interest.
    max_clusters : int, optional
        Maximum number of clusters. Default is 20.

    Returns
    -------
    optimal_clusters : int
        Optimal number of clusters for the graph represented by A.
    """
    degrees = np.array(A.sum(axis=1)).flatten()
    D = np.diag(degrees)    # Degree matrix
    L = D - A               # Laplacian matrix

    eigvals, _ = scipy.sparse.linalg.eigsh(L.astype(float), k=max_clusters, which="SM")
    eigengaps = np.diff(eigvals)

    sorted_indices = np.argsort(eigengaps)[::-1]

    optimal_clusters = sorted_indices[0] + 1
    if optimal_clusters == 1 and len(sorted_indices) > 1:
        optimal_clusters = sorted_indices[1] + 1
    return optimal_clusters


def iterative_spectral_clustering(graph: nx.Graph, max_size: int = 1300, depth: int = 0) -> dict:
    """
    Recursively applies spectral clustering on large clusters.

    Creates a dictionary of clusters with less than `max_size` nodes each.

    Parameters
    ----------
    graph : nx.Graph
        NetworkX Graph object produced by `prep_graph_networkx()`.
    max_size : int, optional
        The maximum number of nodes allowed per cluster. If a cluster exceeds this size,
        it will be recursively split into smaller clusters. All clusters contain fewer than
        this many nodes. Default is 1300.
    depth : int, optional
        Current recursion depth. Default is 0.

    Returns
    -------
    clusters : dict
        Dictionary mapping a cluster label to the NetworkX Graph object. Some entries contain
        sub-dictionaries with more clusters, while others contain subgraphs that are not further split.
    """
    A = nx.adjacency_matrix(graph).toarray()
    n_clusters = optimal_spectral_clustering(A)
    clustering = SpectralClustering(n_clusters=n_clusters, affinity="precomputed").fit_predict(A)

    clusters = {}
    for label in np.unique(clustering):
        nodes = [node for i, node in enumerate(graph.nodes()) if clustering[i] == label]
        subgraph = graph.subgraph(nodes).copy()
        if len(nodes) > max_size:
            clusters[label] = iterative_spectral_clustering(subgraph, max_size, depth + 1)
        else:
            clusters[label] = subgraph
    return clusters
