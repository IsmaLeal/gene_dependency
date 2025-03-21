import os
import torch
import logging
import argparse
import numpy as np
import pandas as pd
import networkx as nx
from tqdm import tqdm
from node2vec import Node2Vec
import HNHN.hypergraph as hnhn
from itertools import combinations
from collections import defaultdict
from multiprocessing import cpu_count
from concurrent.futures import ThreadPoolExecutor, as_completed
from hypergraphx.measures.eigen_centralities import power_method
from general.network_construction import (create_graph_networkx,
                                          extract_subgraphs, create_subhypergraphs,
                                          compute_betweenness_contributions)

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


def obtain_graph_metrics(g) -> pd.DataFrame:
    """
    Computes various topological metrics for each subgraph extracted from `g`.
    
    This function first calls `extract_subgraphs(g)` to decompose `g` into its subgraphs,
    then computes the following metrics for each node from each subgraph:
    
    - **Degree centrality**: Scaling of the number of neighbours of the node.
    - **Betweenness centrality**: Quantifies how frequently a node lies on shortest paths
      between pairs of nodes.
    - **Closeness centrality**: Related to the average distance from a node to all other
      nodes in a graph.
    - **Eigenvector centrality**: Measures a node's importance based on its neighbours'
      importance.
    - **Clustering coefficient**: Measures the level of union in the neighbourhood of
      the node.
    - **Bottleneck centrality**: Quantifies how often a node acts as a critical connection
      in shortest path trees accross the network.
    
    All metrics are computed using **NetworkX**, except for the **Bottleneck coefficient**,
    which is implemented in the function `bottleneck()`.
    
    Parameters
    ----------
    g : nx.Graph
        NetworkX graph representing gene dependency, built from correlations extracted
        from the CRISPR dataset. It can be obtained using `create_graph_networkx` from
        `../general/network_construction.py`.
    
    Returns
    -------
    metrics_df : pd.DataFrame
        Dataframe with the computed metrics for each node from each subgraph (i.e., for
        each gene-pathway pair). The columns are:
        
        - `Gene`: The node index.
        - `Gene name` The gene associated with the node.
        - `Pathway`: The name of the pathway in which the gene is involved.
        - `Degree`: The degree centrality of the node in the subgraph.
        - `Betweenness`: Betweenness centrality of the node.
        - `Closeness`: Closeness centrality of the node.
        - `Eigenvector`: Eigenvector centrality of the node.
        - `Clustering coef`: Local clustering coefficient of the node.
        - `Bottleneck`: Bottleneck coefficient, as defined in [1].
    
    Notes
    -----
    - **Degree centrality**: The degree centrality of a node v is defined as:
    
        c_D(v) = deg(v) / (|V| - 1),

      where:
      - `deg(v)`: The degree of node v (number of neighbours).
      - `|V|`: The order, or total number of nodes, of the graph.
    
    - **Betweenness centrality**: The betweenness centrality of a node v is given by:
    
        b(v) = (2 / ((|V| - 1)(|V| - 2))) * Σ_{s≠t≠v} (σ_st(v) / σ_st),
    
      where:
      - `σ_st`: The number of shortest paths between nodes s and t.
      - `σ_st(v)`: The number of shortest paths between s and t that pass through node v.
      - `|V|`: The order of the graph.
    
    - **Closeness centrality**: The closeness centrality of a node v is defined as:
    
        c(v) = (|V| - 1) / Σ_{u≠v} d_uv,
    
      where:
      - `d_uv`: The shortest path distance between nodes u and v.
      - `|V|`: The order of the graph.
    
    - **Eigenvector centrality**: The eigenvector centrality of a node v is given by the v-th
      entry of the normalised dominant eigenvector of the adjacency matrix.
    
    - **Clustering coefficient**: The local clustering coefficient of a node v is defined as
      the proportion of triangles in the neighbourhood of v:
    
        clust(v) = (2 * |{e_ij: i,j ∈ N(v), e_ij ∈ E}|) / (k_v(k_v - 1)),
      
      where:
      - `N(v)`: The set of neighbours of v.
      - `k_v` = |N(v)|: The degree of node v.
      - `e_ij`: An edge connecting nodes i and j.
      - `E`: The set of edges of the graph.
    
    - **Bottleneck coefficient**: The bottleneck coefficient is defined as:
    
        bn(v) = Σ_{u ∈ V} p_v(u),
    
      where:
      - `p_v(u)` = 1 if more than |V|/4 shortest paths from the shortest path tree rooted
        at node u pass through v; otherwise, `p_v(u)` = 0.
    
    References
    ----------
    [1] C. -H. Chin, S. -H. Chen, H. -H. Wu, C. -W. Ho, M. -T. Ko, and C. -Y. Lin,
        *"cytoHubba: identifying objects and sub-networks from complex interactome"*,
        *BMC Systems Biology*, vol. 8, no. S11, 2014.
        DOI: [10.1186/1752-0509-8-S4-S11](https://doi.org/10.1186/1752-0509-8-S4-S11).

    Examples
    --------
    >>> g = create_graph_networkx(0.2)
    >>> graph_metrics_df = obtain_graph_metrics(g)
    >>> graph_metrics_df.to_csv("path_to_save.csv", index=False)
    """
    # Decompose gene dependency graph into pathways and clusters
    subgraphs = extract_subgraphs(g)

    # Get gene names
    gene_names = pd.read_csv("../datasets/names.txt", header=None)[0]

    i = 0
    metrics = {}        # `metrics` stores the computed metrics for each pathway
    for pathway, subgraph in tqdm(subgraphs.items()):
        logging.info(f"Processing pathway {pathway} ({subgraph.number_of_nodes()} nodes and {subgraph.number_of_edges()} edges)")

        values = {}     # `values` stores the computed metrics for every node in `subgraph`

        if subgraph.number_of_nodes() <= 2:
            continue

        # Degree centrality
        values['degree'] = nx.degree_centrality(subgraph)
        # Betweenness centrality
        values['betweenness'] = nx.betweenness_centrality(subgraph)
        # Closeness centrality
        values['closeness'] = nx.closeness_centrality(subgraph)
        # Eigenvector centrality
        values['eigenvector'] = nx.eigenvector_centrality(subgraph, max_iter=5000, tol=1e-6)
        # Local clustering coefficient
        values['clustering'] = nx.clustering(subgraph)
        # Bottleneck coefficient
        values['bottleneck'] = bottleneck(subgraph)

        metrics[pathway] = values
        i += 1

    data = []       # `data` is a list of dictionaries, one for the metrics of each gene-pathway pair
    for pathway, metrics in tqdm(metrics.items()):
        for node in metrics['degree']:
            data.append({
                'Gene': node,
                'Gene name': gene_names[node],
                'Pathway': pathway,
                'Degree': metrics['degree'][node],
                'Betweenness': metrics['betweenness'][node],
                'Closeness': metrics['closeness'][node],
                'Eigenvector': metrics['eigenvector'][node],
                'Clustering coef': metrics['clustering'][node],
                'Bottleneck': metrics['bottleneck'][node]
            })

    metrics_df = pd.DataFrame(data)
    metrics_df = metrics_df.sort_values(by=['Gene', 'Pathway'], ascending=True)
    return metrics_df


def obtain_hypergraph_metrics(hypergraphs) -> pd.DataFrame:
    """
    Compute various topological metrics for each hypergraph in the input dictionary `hypergraphs`.

    This function iterates over a collection of hypergraphs, each associated with a pathway. For each
    hypergraph, it first computes a star expansion, which is a bipartite graph representation derived
    from the hypergraph's incidence matrix. Then, it calculates several metrics for each node of the
    hypergraph by evaluating properties from the corresponding star expansion graph.

    Some metrics have a custom implementation. This is because the metrics computed using NetworkX
    account for all the nodes in the input graph, as usual. However, for the star expansion, a set of
    nodes represents the hyperedges, and another set of nodes represents the hypergraph nodes (genes).
    Hence, the nodes representing hyperedges should be excluded from all computations and shortest paths.

    The following metrics are computed:

    - **Degree centrality**: The normalised degree of each hypergraph node. The raw degree is obtained
      from the hypergraph's degree sequence and then scaled by the factor 1 / (number of nodes - 1).
    - **Betweenness centrality**: Quantifies how frequently a node lies on the shortest paths between
      pairs of nodes in the star expansion network. A custom implementation is used to iterate over all
      pairs of hypergraph nodes and count the fraction of shortest paths passing through each node. A
      scaling factor is applied, and the obtained values are re-mapped from the star expansion node
      indices to the original hypergraph node labels.
    - **Closeness centrality**: Measures how close a node is to all other nodes in the star expansion. A
      custom implementation computes the shortest path lengths from each node to every other node and
      computes closeness as the (number of reachable nodes minus one) divided by the total distance. Only
      those nodes representing hypergraph nodes are accounted for. The final closeness values are re-mapped
      to correspond to the original hypergraph nodes.
    - **Eigenvector centrality**: Computed using the power method on the adjacency matrix of the star
      expansion. The adjacency matrix is constructed by combining the incidence matrix and zero blocks,
      so that the hypergraph is represented as a bipartite graph. The entries of the resulting dominant
      eigenvector associated to hypergraph nodes give the eigencentrality values (after normalising), which
      are re-mapped to hypergraph node labels.
    - **Clustering coefficient**: An ad hoc clustering coefficient is computed using properties of the
      hypergraph:
      - For each node, the number of potential neighbour pairs is computed based on the number of neighbours
        provided by the hypergraph.
      - The actual "clustering" is defined as the ratio between the number of incident hyperedges and the
        computed number of neighbours.
    - **Bottleneck coefficient**: A bottleneck coefficient is computed via another function called
      `bottleneck_hypergraph()`. This metric is designed to capture the role of a node as a critical connector
      in the network. The computed values are then remapped to the original hypergraph node labels.

    Parameters
    ----------
    hypergraphs : dict
        A dictionary where the keys are the different pathways or clusters and the values are the corresponding
        HypergraphX Hypergraph objects.

    Returns
    -------
    metrics_df : pd.DataFrame
        DataFrame with the computed metrics for each node from each hypergraph (i.e., for each gene-pathway
        pair). Each row corrresponds to a gene-pathway pair and the columns are:
        - `Gene`: The node index.
        - `Gene name` The gene associated with the node.
        - `Pathway`: The name of the pathway in which the gene is involved.
        - `Degree`: The degree centrality of the node in the subgraph.
        - `Betweenness`: Betweenness centrality of the node.
        - `Closeness`: Closeness centrality of the node.
        - `Eigenvector`: Eigenvector centrality of the node.
        - `Clustering coef`: Local clustering coefficient of the node.
        - `Bottleneck`: Bottleneck coefficient, as defined in [1].

    Notes
    -----
    - Star expansion:
      The hypergraph is converted into a bipartite graph (star expansion) by using its incidence matrix.
      In this representation, nodes representing original hypergraph elements and nodes representing hyperedges
      form two disjoint sets, and the connection between them is given by the incidence of a node in a hyperedge.

    Examples
    --------
    >>> hypergraphs = create_subhypergraphs()
    >>> HG_metrics_df = obtain_hypergraph_metrics(hypergraphs)
    >>> HG_metrics_df.to_csv("path_to_save.csv", index=False)
    """
    # Get gene names
    gene_names = pd.read_csv("../datasets/names.txt", header=None)[0]

    i = 0
    metrics = {}        # `metrics` stores the computed metrics for pathway

    for pathway, h in tqdm(hypergraphs.items()):
        logging.info(f'\nMetrics of {pathway} ({h.num_nodes()} nodes and {h.num_edges()} hyperedges)')
        values = {}     # `values` stores the computed metrics for every node in the current hypergraph `h`

        # Degree sequence (raw degree counts)
        deg_seq = h.degree_sequence()

        # Compute adjacency matrix of star expansion (bipartite graph)
        inc, mapping = h.incidence_matrix(return_mapping=True)
        inc_dense = inc.toarray()
        n_nodes = h.num_nodes()
        n_edges = h.num_edges()

        # Create block matrices for nodes and hyperedges, then combine them into the adjacency matrix A
        zeros_nodes = np.zeros((n_nodes, n_nodes))
        zeros_edges = np.zeros((n_edges, n_edges))
        A = np.vstack((
            np.hstack((zeros_nodes, inc_dense)),
            np.hstack((inc_dense.T, zeros_edges))
        ))
        # Convert the adjacency matrix to a NetworkX graph
        star_expansion = nx.from_numpy_array(A)

        # Scale degree
        values['degree'] = {k: v / (n_nodes - 1) for k, v in deg_seq.items()}

        # ------------------------------------------------------------------------
        # Betweenness centrality (custom implementation):
        # For each pair of hypergraph nodes (as represented in the star expansion), all shortest paths are computed
        # using `calculate_shortest_paths()`. For every intermediate node on a shortest path, a contribution of
        # 1/total_paths is added.
        # Finally, the betweenness for each node is scaled by the normalizing factor.
        # Only the first n_nodes of the star expansion correspond to the original hypergraph nodes.
        betweenness = {idx: 0.0 for idx, node in enumerate(star_expansion.nodes()) if idx < n_nodes}

        # Extract indices corresponding to hypergraph nodes (exclude nodes representing hyperedges)
        hg_nodes = [idx for idx, node in enumerate(star_expansion.nodes()) if idx < n_nodes]

        max_threads = min(50, cpu_count())
        with ThreadPoolExecutor(max_workers=max_threads) as executor:
            futures = {executor.submit(compute_betweenness_contributions, star_expansion, hg_nodes, s, t): (s, t)
                       for s, t in tqdm(combinations(hg_nodes, 2), total=len(hg_nodes) * (len(hg_nodes) - 1) // 2)}

            for future in tqdm(as_completed(futures), total=len(futures)):
                result = future.result()
                for node, value in result:
                    betweenness[node] += value

        # Apply a normalising scaling factor based on the number of possible node pairs
        scale = 1 / ((n_nodes - 1) * (n_nodes - 2) / 2) if n_nodes > 2 else 1
        betweenness = {k: v * scale for k, v in betweenness.items()}
        betweenness = {mapping[k]: betweenness[k] for k in range(n_nodes)}
        values['betweenness'] = betweenness

        # ------------------------------------------------------------------------
        # Closeness centrality (custom implementation):
        # For each hypergraph node (from the star expansion), compute the sum of shortest path lengths
        # to all other hypergraph nodes. Closeness is then defined as (number of other nodes) divided by total distance.
        closeness = {node: 0.0 for idx, node in enumerate(star_expansion.nodes()) if idx < n_nodes}
        hg_nodes = [node for idx, node in enumerate(star_expansion.nodes()) if idx < n_nodes]

        for v in hg_nodes:
            shortest_paths_lenghts = nx.single_source_shortest_path_length(star_expansion, v)
            total_distance = 0
            for node in hg_nodes:
                if node != v:
                    try:
                        total_distance += shortest_paths_lenghts[node]
                    except KeyError:
                        continue
            if total_distance > 0:
                closeness[v] = (len(hg_nodes) - 1) / total_distance

        closeness = {mapping[k]: closeness[k] for k in range(n_nodes)}
        values['closeness'] = closeness

        # Eigencentrality via power method
        eigencentrality = power_method(A, max_iter=10000, tol=1e-8)
        values['eigencentrality'] = {mapping[k]: eigencentrality[k] for k in range(n_nodes)}

        # ------------------------------------------------------------------------
        # Clustering coefficient (custom implementation):
        # For each hypergraph node, compute:
        #   - n_neighbours: 2^(neighbours+1) - neighbours - 2, representing the maximum
        #     possible number of neighboring pairs.
        #   - n_hyperedges: the actual number of hyperedges incident to the node.
        # The clustering coefficient is defined as the ratio between the number of hyperedges and n_neighbours.
        node_degrees = {node: len(h.get_neighbors(node)) for node in h.get_nodes()}
        n_neighbours = {
            node: 2 ** (deg + 1) - deg - 2 for node, deg in node_degrees.items()
        }
        n_hyperedges = {node: len(h.get_incident_edges(node)) for node in h.get_nodes()}

        clustering = {
            node: (n_hyperedges[node] / n_neighbours[node]) if n_neighbours[node] != 0 else 0
            for node in h.get_nodes()
        }

        clustering = {mapping[k]: clustering[node] for k, node in enumerate(h.get_nodes())}
        values['clustering'] = clustering

        # ------------------------------------------------------------------------
        # Bottleneck coefficient:
        # Use `bottleneck_hypergraph()` function.
        bn = bottleneck(star_expansion, n_nodes)
        bn = {mapping[k]: bn[k] for k in range(n_nodes)}
        values['bn'] = bn

        metrics[pathway] = values
        i += 1

    data = []
    for pathway, measures in tqdm(metrics.items()):
        for node in measures['degree']:
            data.append({
                'Gene': node,
                'Gene name': gene_names[node],
                'Pathway': pathway,
                'Degree': measures['degree'][node],
                'Betweenness': measures['betweenness'][node],
                'Closeness': measures['closeness'][node],
                'Eigencentrality': measures['eigencentrality'][node],
                'Clustering coef.': measures['clustering'][node],
                'Bottleneck': measures['bn'][node]
            })

    metrics_df = pd.DataFrame(data)
    metrics_df = metrics_df.sort_values(by=['Gene', 'Pathway'], ascending=True)
    return metrics_df


def obtain_graph_embeddings(g, dim=20, walk_length=30, num_walks=200, workers=4) -> pd.DataFrame:
    """
    Computes Node2Vec embeddings for each subgraph extracted from the input graph.

    This function applies the Node2Vec algorithm to generate low-dimensional representations
    of nodes in each subgraph of `g`, helping capture structural information in an unsupervised
    manner.

    Parameters
    ----------
    g : nx.Graph
        NetworkX input graph representing gene dependencies.
    dim : int, optional
        The number of dimensions for the embedding vectors. Default is 20.
    walk_length : int, optional
        The length of each random walk used in Node2Vec. Default is 30.
    num_walks : int, optional
        The number of random walks per node. Default is 200.
    workers : int, optional
        The number of parallel workers for Node2Vec computation. Default is 4.

    Returns
    -------
    df : pd.DataFrame
        Pandas DataFrame containing the embeddings for each gene-pathway pair.
        Columns are:
        - `Gene`: Node index in the original graph `g`.
        - `Gene name`: Corresponding gene name.
        - `Pathway`: The pathway or cluster for the gene-pathway pair.
        - Embedding dimensions (e.g., `0`, `1`, ..., `dim-1`).

    Notes
    -----
    - Node2Vec is a graph embedding method that uses biased random walks to capture both
      local and global graph structures. The embeddings are learned via a Skip-Gram model,
      similar to word embeddings in NLP.
    - This function extracts subgraphs, applies Node2Vec separately to each, and concatenates
      the results.

    Raises
    ------
    FileNotFoundError
        If `../datasets/names.txt` is not found.
    ValueError
        If the input graph `g` is empty or contains no valid subgraphs.
    RuntimeError
        If Node2Vec encounters an issue during model training.

    Examples
    --------
    >>> g = create_graph_networkx(0.2)
    >>> graph_embeddings_df = obtain_graph_embeddings(g, dim=40)
    >>> graph_embeddings_df.to_csv("path_to_save.csv", index=False)
    """
    if g is None or g.number_of_nodes() == 0:
        raise ValueError("Input graph `g` is empty or None.")
    subgraphs = extract_subgraphs(g)
    if not subgraphs:
        raise ValueError("An error occurred while extracting the subgraphs.")

    names_path = "../datasets/names.txt"
    if not os.path.exists(names_path):
        raise FileNotFoundError(f"Missing file: {names_path}. Ensure the dataset is available.")
    names = pd.read_csv(names_path, header=None)[0]
    idx_to_name = {idx: name for idx, name in enumerate(names)}

    dfs = []
    for pathway, subgraph in tqdm(subgraphs.items(), desc="Computing Node2Vec embeddings"):
        logging.info(f"Processing pathway: {pathway} ({subgraph.number_of_nodes()} nodes, {subgraph.number_of_edges()} edges.")

        try:
            node2vec = Node2Vec(subgraph, dimensions=dim, walk_length=walk_length, num_walks=num_walks, workers=workers)
            model = node2vec.fit(window=10, min_count=1, batch_words=4)
        except Exception as e:
            raise RuntimeError(f"Node2Vec encountered an error while processing pathway {pathway}: {e}")

        embeddings = model.wv

        embedding_df = pd.DataFrame([embeddings[str(node)] for node in subgraph.nodes()])
        embedding_df["Gene"] = list(subgraph.nodes())
        embedding_df["Pathway"] = pathway
        dfs.append(embedding_df)

    df = pd.concat(dfs, ignore_index=True)

    df["Gene name"] = df["Gene"].map(idx_to_name)

    cols_to_move = ["Gene", "Gene name", "Pathway"]
    df = df[cols_to_move + [col for col in df.columns if col not in cols_to_move]]

    df = df.sort_values(by=["Gene", "Pathway"]).reset_index().iloc[:, 1:]

    return df


def obtain_hypergraph_embeddings(hypergraphs: dict, dim: int=128, dropout_p: float=0.3) -> pd.DataFrame:
    """
    Computes node embeddings for hypergraphs using the HNHN model [1].

    This function extracts embeddins from hypergraphs using the HNHN model, which learns
    node representations with a neural network architecture designed for hypergraphs. The
    embeddings are returned as a DataFrame.

    Parameters
    ----------
    hypergraphs : dict
        A dictionary where the keys are the different pathways or clusters and the values are the corresponding
        HypergraphX Hypergraph objects.
    dim : int, optional
        Dimension of the hidden embeddings in the model. Default is 128.
    dropout_p : float, optional
        Dropout probability for the neural network layers. Default is 0.3.

    Returns
    -------
    df : pd.DataFrame
        DataFrame containing the embeddings for each gene-pathway pair. The columns are:
        - `Gene`: Node in the hypergraph.
        - `Gene name`: Name of the corresponding gene.
        - `Pathway`: The corresponding pathway.
        - Embedding dimensions (e.g., `0`, `1`, ...).

    Raises
    ------
    FileNotFoundError
        If the `../datasets/names.txt` file is missing.
    RuntimeError
        If HNHN fails to generate embeddings.

    References
    ----------
    [1] Y. Dong, W. Sawin, and Y. Bengio, *"HNHN: Hypergraph Networks with Hyperedge Neurons"*,
        *arXiv prepring*, arXiv:2006.12278v1, 2020.
        DOI: [10.48550/arXiv.2006.12278](https://doi.org/10.48550/arXiv.2006.12278).

    Examples
    --------
    >>> hypergraphs = create_subhypergraphs()
    >>> hg_embeddings_df = obtain_hypergraph_metrics(hypergraphs, dim=40)
    >>> hg_embeddings_df.to_csv("path_to_save.csv", index=False)
    """
    logging.info("Loading pathway mappings and preparing hypergraphs...")

    names_path = "../datasets/names.txt"
    if not os.path.exists(names_path):
        raise FileNotFoundError(f"Missing required file: {names_path}.")

    names = pd.read_csv(names_path, header=None)[0]
    idx_to_name = {idx: name for idx, name in enumerate(names)}

    embeddings_list = []

    # Iterate over the hypergraphs
    for pathway, h in tqdm(hypergraphs.items(), desc="Processing hypergraphs"):
        hyperedges = h.get_edges()
        nodes = h.get_nodes()
        nv, ne = len(nodes), len(hyperedges)
        logging.info(f"Processing pathway {pathway} ({nv} nodes, {ne} hyperedges).")

        node_to_idx = {node: idx for idx, node in enumerate(nodes)}

        # Build adjacency data for HNHN
        vertex_idx, hyperedge_idx, paper_author = [], [], []
        for idx, hyperedge in enumerate(hyperedges):
            for node in hyperedge:
                vertex_idx.append(node_to_idx[node])
                hyperedge_idx.append(idx)
                paper_author.append([node_to_idx[node], idx])

        # Convert adjacency data to tensors
        vertex_idx = torch.tensor(vertex_idx, dtype=torch.int64)
        hyperedge_idx = torch.tensor(hyperedge_idx, dtype=torch.int64)
        paper_author = torch.tensor(paper_author, dtype=torch.int64)

        # Initialise weights
        vertex_weights = torch.ones(len(nodes), 1, dtype=torch.float32)
        hyperedge_weights = torch.ones(len(hyperedges), 1, dtype=torch.float32)

        # Model parameters
        args = argparse.Namespace(
            paper_author=paper_author, n_hidden=dim, predict_edge=False,
            n_cls=2, edge_linear=False, input_dim=nv, n_layers=2, dropout_p=dropout_p,
            v_reg_weight=torch.Tensor([1.0]), e_reg_weight=torch.Tensor([1.0]),
            v_reg_sum=torch.Tensor([1.0]), e_reg_sum=torch.Tensor([1.0]),
            nv=nv, ne=ne
        )

        # Initialise model
        try:
            model = hnhn.Hypergraph(
                vertex_idx, hyperedge_idx, nv, ne, vertex_weights,
                hyperedge_weights, args
            )
            v_init = torch.randn(nv, args.input_dim)
            e_init = torch.zeros(ne, dim)

            vertex_embeddings, _, _ = model(v_init, e_init)
        except Exception as e:
            logging.error(f"Error processing pathway {pathway}: {e}.")
            continue

        # Convert to DataFrame
        embedding_df = pd.DataFrame(vertex_embeddings.detach().numpy())
        embedding_df["Gene"] = nodes
        embedding_df["Pathway"] = pathway

        embeddings_list.append(embedding_df)

    if not embeddings_list:
        raise RuntimeError("No embeddings were successfully computed.")

    df = pd.concat(embeddings_list, ignore_index=True)
    df["Gene name"] = df["Gene"].map(idx_to_name)

    # Reorder columns
    cols_to_move = ["Gene", "Gene name", "Pathway"]
    df = df[cols_to_move + [col for col in df.columns if col not in cols_to_move]]
    df = df.sort_values(by=["Gene", "Pathway"]).reset_index(drop=True)

    return df


def bottleneck(graph, n_nodes=None) -> dict:
    """
    Computes the bottleneck coefficient for every node in a graph or hypergraph (via star expansion).

    This function generalizes the bottleneck coefficient calculation to both graphs and hypergraphs.
    For hypergraphs, it assumes the input `graph` is the **star expansion** (bipartite graph representation),
    where the first `n_nodes` correspond to hypergraph nodes and the remaining nodes represent hyperedges.

    The bottleneck coefficient of a node is the number of shortest path trees in which the node appears
    as an intermediary, provided that it is visited in at least `n_nodes / 4` shortest paths [1].

    Parameters
    ----------
    graph : nx.Graph
        A NetworkX graph, either a standard graph or the star expansion of a hypergraph.
    n_nodes : int
        The number of original nodes in the hypergraph (before expansion). This ensures that hyperedge nodes
        are excluded from computations.

    Returns
    -------
    bottleneck : dict
        Dictionary mapping (hyper)graph node indices to their bottleneck coefficient.

    Notes
    -----
    - For hypergraphs, only the **first `n_nodes`** are considered to avoid counting artificial hyperedge nodes.
    - The bottleneck coefficient measures how often a node acts as a critical bridge in shortest path trees.
    - The function ensures that nodes are only counted within their connected component.

    References
    ----------
    [1] C. -H. Chin, S. -H. Chen, H. -H. Wu, C. -W. Ho, M. -T. Ko, and C. -Y. Lin,
        *"cytoHubba: identifying objects and sub-networks from complex interactome"*,
        *BMC Systems Biology*, vol. 8, no. S11, 2014.
        DOI: [10.1186/1752-0509-8-S4-S11](https://doi.org/10.1186/1752-0509-8-S4-S11).
    """
    if n_nodes == None:
        n_nodes = len(graph.nodes())
    bottleneck = {node: 0.0 for idx, node in enumerate(graph.nodes()) if idx < n_nodes}
    nodes = [node for idx, node in enumerate(graph.nodes()) if idx < n_nodes]

    for root in nodes:
        shortest_tree = nx.single_source_shortest_path(graph, root)
        # Count number of paths passing through each node
        path_counts = defaultdict(int)
        for target, path in shortest_tree.items():
            if target == root:
                continue
            for node in path:
                if node != root and node != target and node in nodes:
                    path_counts[node] += 1

        # Ensure bottleneck is calculated only within connected components
        components = list(nx.connected_components(graph))
        for component in components:
            for node, count in path_counts.items():
                if (count > n_nodes / 4) and (node in component):
                    bottleneck[node] += 1

    return bottleneck


if __name__ == "__main__":
    # Create graphs and hypergraphs
    g = create_graph_networkx(0.2)
    hypergraphs = create_subhypergraphs()

    # Compute topological metrics
    g_metrics = obtain_graph_metrics(g)
    hg_metrics = obtain_hypergraph_metrics(hypergraphs)

    # Compute embeddings
    g_embeddings = obtain_graph_embeddings(g)
    hg_embeddings = obtain_hypergraph_embeddings(hypergraphs)

    # Save results
    g_metrics.to_csv("./features/graph_metrics.csv", index=False)
    hg_metrics.to_csv("./features/hypergraph_metrics.csv", index=False)

    g_embeddings.to_csv("./features/graph_embeddings.csv", index=False)
    hg_embeddings.to_csv("./features/hypergraph_embeddings.csv", index=False)
