import os
import ast
import logging
import numpy as np
from tqdm import tqdm
import networkx as nx
import graph_tool as gt
from scipy.sparse import csr_matrix
from . utils import clean_col_names, safe_read_csv

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


def create_graph_networkx(threshold: float) -> nx.Graph:
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
    data_path = "../datasets/"

    corrs = safe_read_csv(os.path.join(data_path, "abs_corrs.csv"), delimiter=",", index_col=0)
    corrs /= np.max(corrs.values)

    gene_names = np.array([clean_col_names(col) for col in corrs.columns])  # Remove gene IDs

    # Use sparse matrix for adjacency matrix
    A = csr_matrix((corrs.values > threshold).astype(np.int8))
    g = nx.from_scipy_sparse_array(A)

    nx.set_node_attributes(g, {i: gene_names[i] for i in range(len(gene_names))}, "name")  # Add gene names
    return g


def convert_nx_to_gt(graph: nx.Graph) -> gt.Graph:
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
    node_map = {}  # Global to local node index map

    for node in graph.nodes():
        v = gt_graph.add_vertex()
        node_map[node] = v
        gt_graph.vp.node_label[v] = node

    for edge in graph.edges(data=True):
        u, v, data = edge
        gt_graph.add_edge(node_map[u], node_map[v])

    return gt_graph


def create_subgraphs(g: nx.Graph) -> dict:
    """
    Returns a dictionary mapping protein pathways to NetworkX graphs.

    Converts pathway groupings from `../datasets/pathway_mapping.csv`
    into NetworkX subgraphs. Excludes subgraphs without edges or with
    less than three nodes.

    Parameters
    ----------
    g : nx.Graph
        Graph object created using `prep_graph_networkx()` from CRISPR data.

    Returns
    -------
    subgraphs : dict
        Dictionary mapping pathway name to NetworkX Graph object.
    """
    # Load pathway-to-nodes mapping
    file_path = "../datasets/pathway_mapping.csv"
    pathway_mapping = safe_read_csv(file_path)
    if not pathway_mapping:
        raise FileNotFoundError(
            f"The file {file_path} is missing."
            f"Run `map_pathway_to_nodes()` from `../data_processing/pathway_mapping.py` first."
        )

    # Iterate over pathways
    subgraphs = {}
    for _, row in tqdm(pathway_mapping.iterrows(), desc="Creating subgraphs"):
        pathway = row["Pathway"]
        nodes = ast.literal_eval(row["Nodes"])
        subgraph = g.subgraph(nodes)
        if (subgraph.number_of_edges() > 0) & (subgraph.number_of_nodes() > 2):
            subgraphs[pathway] = subgraph
        else:
            pass
    return subgraphs
