import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, List
import networkx as nx
import graph_tool.all as gt
from graph_tool.inference import CliqueState
from sklearn.cluster import SpectralClustering
from tqdm import tqdm
import time
import csv
import scipy.sparse
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Configuration
data_path = "../datasets/"
threshold = 0.2
num_clusters = 3
mcmc_sweeps = 1000

# Preprocessing functions
def clean_col_names(col: str) -> list[str]:
    """Extracts gene names from a label formatted as 'GeneName (GeneID)'."""
    return col.split(' (')[0]

def prep_graph_networkx(threshold: float = threshold) -> nx.Graph:
    """
    Loads correlation data, applies threshold, and returns a NetworkX graph.

    This function computes the adjacency matrix from the given threshold and returns
    a NetworkX graph together with a list of the gene names (with the gene IDs removed).

    Parameters
    ----------
    threshold : float
        Value between 0 and 1. Gene pairs with an absolute correlation above the given
        threshold will be represented as nodes that share an edge.

    Returns
    -------
    g : nx.Graph
        Graph object representing gene dependency.
    """
    logging.info("Preparing graph...")
    corrs = pd.read_csv(os.path.join(data_path, "abs_corrs.csv"), delimiter=",", index_col=0)
    corrs /= np.max(corrs.values)
    gene_names = np.array([clean_col_names(col) for col in corrs.columns])  # Remove gene IDs
    A = (corrs.values > threshold).astype(np.int8)  # Adjacency matrix
    g = nx.from_numpy_array(A)
    nx.set_node_attributes(g, {i: gene_names[i] for i in range(len(gene_names))}, "name")   # Add gene names
    return g

def map_pathway_to_nodes() -> pd.DataFrame:
    """
    Maps each pathway to the corresponding nodes.

    This function returns a dataframe mapping each pathway to the nodes (names & indices) it contains.
    Includes two clusterings apart from the pathways extracted from UniProt:
    - "Unmapped": includes all genes with either NaN pathways or belonging to a single-gene pathway.
    - "Unknown": includes all genes from CRISPR not belonging to the UniProt dataset.
    """
    genes = pd.read_csv(os.path.join(data_path, "names.txt"), header=None)[0]
    gene_to_index = {name: idx for idx, name in enumerate(genes)}

    dfs = []
    for i in range(4):
        df = pd.read_excel(os.path.join(data_path, f"uniprot/uniprot{i+1}.xlsx"))
        dfs.append(df)
    df = pd.concat([i for i in dfs], ignore_index=True)
    df = df[df["Reviewed"] == "reviewed"].reset_index().iloc[:, 1:]
    df = df[["From", "Reactome"]]  # Select columns with gene name and pathway name

    # Remove NaN values and add them to the unmapped group
    unmapped = df[df["Reactome"].isna()]
    unmapped_list = list(np.unique(unmapped["From"].values))
    df = df.dropna(subset=["Reactome"]).reset_index().iloc[:, 1:]

    # Create an extended dataframe with a 1 to 1 mapping from every gene to all the pathways it belongs to
    name_list, pathway_list = [], []
    for idx, row in df.iterrows():
        pathways = str(df.iloc[idx, 1])
        pathways = [pathway for pathway in pathways.split(";") if pathway != ""]
        for pathway in pathways:
            name_list.append(df.iloc[idx, 0])
            pathway_list.append(pathway)
    extended = pd.DataFrame({"Genes": name_list, "Pathway": pathway_list})

    # Group by pathway
    pathway_to_genes = extended.groupby("Pathway")["Genes"].agg(list).reset_index()
    # We'll take single-gene pathways to be in the same classification as those genes without a pathway
    unmapped = pathway_to_genes[[len(genes) < 2 for genes in pathway_to_genes["Genes"].values]]
    unmapped_list.extend([gene[0] for gene in np.unique(unmapped["Genes"].values)])
    # Classify the rest of the CRISPR genes as unknown
    unknown_list = [gene for gene in genes if (not gene in extended["Genes"].values)
                    & (not gene in unmapped_list)]
    pathway_to_genes = pathway_to_genes[
                           [len(genes) > 1 for genes in pathway_to_genes["Genes"].values]].reset_index().iloc[:, 1:]
    # Add "unmapped" and "unknown" to the DF
    pathway_to_genes.loc[len(pathway_to_genes)] = pd.Series({"Pathway": "No pathway", "Genes": unmapped_list})
    pathway_to_genes.loc[len(pathway_to_genes)] = pd.Series({"Pathway": "Unknown pathway", "Genes": unknown_list})

    # Add node indices
    nodes = []
    for idx, row in pathway_to_genes.iterrows():
        pathway_nodes = [gene_to_index[gene] for gene in row["Genes"] if gene in gene_to_index]
        nodes.append(pathway_nodes)
    pathway_to_genes["Nodes"] = nodes

    # Remove duplicate pathways consisting of a set of nodes already present in the DF
    df = pathway_to_genes.drop_duplicates(subset="Nodes", keep="first").reset_index().iloc[:, 1:]
    df["num_genes"] = df["Nodes"].apply(len)

    return df


def create_subgraphs(g: nx.Graph, pathway_df: pd.DataFrame) -> dict:
    """
    Converts pathway groupings into actual NetworkX subgraphs.

    Parameters
    ----------
    g : nx.Graph
        Graph object created using `prep_graph_networkx()` from CRISPR data.
    pathway_df : pd.DataFrame
        Dataframe created using `map_pathway_to_nodes()`.

    Returns
    -------
    subgraphs : dict
        Dictionary mapping pathway name to NetworkX Graph object.
    """
    subgraphs = {}
    for _, row in tqdm(pathway_df.iterrows(), desc="Creating subgraphs"):
        pathway = row["Pathway"]
        nodes = row["Nodes"]
        subgraph = g.subgraph(nodes)
        if (subgraph.number_of_edges() > 0) & (subgraph.number_of_nodes() > 2):
            subgraphs[pathway] = subgraph
        else:
            pass
    return subgraphs


def nx_to_gt(graph: nx.Graph) -> gt.Graph:
    """
    Takes NetworkX Graph object and returns a graph-tool Graph.

    Spectral clustering is applied using NetworkX, but the hyperedge inference
    is implemented in graph-tool. Hence, conversion between both is needed. Global
    node indexing is not lost, in order to keep track of the genes.

    Parameters
    ----------
    graph : nx.Graph
        Input NetworkX Graph object.

    Returns
    -------
    gt_graph : graph_tool.Graph
        graph-tool Graph object.
    """
    gt_graph = gt.Graph(directed=False)
    gt_graph.vertex_properties["node_label"] = gt_graph.new_vertex_property("int")  # Local to global node index map
    node_map = {}   # Global to local node index map

    for node in graph.nodes():
        v = gt_graph.add_vertex()
        node_map[node] = v
        gt_graph.vp.node_label[v] = node

    for edge in graph.edges(data=True):
        u, v, data = edge
        gt_graph.add_edge(node_map[u], node_map[v])

    return gt_graph


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
    eigvals, _ = scipy.sparse.linalg.eigsh(L, k=max_clusters, which="SM")
    eigengaps = np.diff(eigvals)
    optimal_clusters = np.argmax(eigengaps) + 1
    return optimal_clusters


def iterative_spectral_clustering(graph: nx.Graph, max_size: int = 1300) -> dict:
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
            clusters[label] = iterative_spectral_clustering(subgraph, max_size)
        else:
            clusters[label] = subgraph
    return clusters


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
        state.mcmc_sweep(niter=1, beta=1)   # Default inverse temperature
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
            gt_subgraph = nx_to_gt(subgraph)
            results[cluster_id] = get_cliques_from_subgraph(gt_subgraph)

    return results


def found_fraction_vs_threshold():
    # Plotting settings
    plt.rc('text', usetex=True)
    plt.rc('font', family='serif')
    plt.rcParams.update({'font.size': 21})

    ns = [5, 6, 7, 8, 85, 9, 95, 96, 97, 98, 99]
    thresholds = [0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.95, 0.96, 0.97, 0.98, 0.99]
    fractions = []
    fractions_wo = []

    for idx, n in enumerate(ns):
        df = pd.read_csv(f'./../1_graph/results_{str(n)}.csv')
        df_filt = df[df['Significant']]
        fraction = 100 * len(df_filt) / len(df)
        df_wo = df[df['Number of genes'] > 6]
        fraction_wo = 100 * len(df_filt) / len(df_wo)
        fractions.append(fraction)
        fractions_wo.append(fraction_wo)

    fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    ax.plot(thresholds, fractions, label='All complexes')
    ax.plot(thresholds, fractions_wo, label='Complexes with $\geq$ 6 nodes')
    ax.set_title('Fraction of found complexes per threshold')
    ax.set_xlabel('Thresholds')
    ax.set_ylabel('Percentage')
    ax.legend()

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    # Load graph
    g = prep_graph_networkx(0.2)

    # Map pathways
    pathway_df = map_pathway_to_nodes()
    pathway_subgraphs = create_subgraphs(g, pathway_df)

    # Initialise dictionary for storing hyperedges
    all_hyperedges = {}
    all_pathways = {}
    all_num_hyperedges = {}
    all_num_nodes = {}

    for pathway, subgraph in pathway_subgraphs.items():
        logging.info(f"Processing pathway: {pathway}")

        # Apply spectral clustering only to pathways large enough to be subdivided
        if subgraph.number_of_nodes() > 1300:
            clusters = iterative_spectral_clustering(subgraph)
        else:
            clusters = {pathway: subgraph}

        hyperedges = extract_hyperedges(clusters)
        all_hyperedges[pathway] = hyperedges
        all_pathways = None

    for pathway, hyperedge_dict in all_hyperedges.items():
        for label, edges in hyperedge_dict.items():
            file_path = f"hyperedges/{pathway}_cluster_{label}.csv"
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
            with open(file_path, mode="w", newline="") as file:
                writer = csv.writer(file)
                for edge in edges:
                    writer.writerow(edge)
