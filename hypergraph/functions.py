import graph_tool.all as gt
from graph_tool.inference import CliqueState

import pandas as pd
import numpy as np
import time
from tqdm import tqdm
import networkx as nx
import scipy.sparse
import scipy.sparse.linalg
from multiprocessing import Pool, Manager
from sklearn.cluster import SpectralClustering
import matplotlib.pyplot as plt


def clean_col_names(col):
    '''Takes a label from the CRISPR dataset which have a structure of 'GeneName (GeneID)'
    and returns only 'GeneName' '''
    return col.split(' (')[0]


def prep_graph(threshold, ranked):
    '''Loads the ranked correlation matrix from 'rank_transf_symm_2.csv', normalises it,
    computes the adjacency matrix from the threshold argument and returns a graph-tool Graph object
    together with a list of the gene names (with the gene IDs removed)'''
    gt.openmp_set_num_threads(10)

    print('Preparing graph...')
    start_time = time.time()
    if ranked:
        corrs = pd.read_csv('../datasets/ranked_corrs.csv', delimiter=',', index_col=0)
    else:
        corrs = pd.read_csv('../datasets/abs_corrs.csv', delimiter=',', index_col=0)
    corrs /= np.max(corrs.values)  # Normalise
    gene_names = np.array([clean_col_names(col) for col in corrs.columns])  # Remove gene IDs

    # Create adjacency matrix A
    A = (corrs.values > threshold).astype(np.int8)

    # Instantiate graph-tool Graph and add nodes & edges based on A
    g = gt.Graph(directed=False)
    g.add_vertex(n=A.shape[0])
    edges = np.transpose(np.nonzero(np.triu(A, 1)))  # Use k=1 to prevent self-interactions
    g.add_edge_list(edges)
    print(f'Data loaded in {time.time() - start_time:.2} seconds.')

    # Add gene names to the nodes
    names = g.new_vertex_property('string')
    for v in g.vertices():
        names[int(v)] = gene_names[int(v)]
    g.vertex_properties['names'] = names

    return g


def prep_graph_networkx(threshold, ranked):
    '''Loads the ranked correlation matrix from 'rank_transf_symm_2.csv', normalises it,
    computes the adjacency matrix from the threshold argument and returns a graph-tool Graph object
    together with a list of the gene names (with the gene IDs removed)'''

    print('Preparing graph...')
    if ranked:
        corrs = pd.read_csv('../datasets/ranked_corrs.csv', delimiter=',', index_col=0)
    else:
        corrs = pd.read_csv('../datasets/abs_corrs.csv', delimiter=',', index_col=0)
    corrs /= np.max(corrs.values)   # Normalise
    gene_names = np.array([clean_col_names(col) for col in corrs.columns])  # Remove gene IDs

    # Create adjacency matrix A
    A = (corrs.values > threshold).astype(np.int8)
    # Create NetworkX graph object from A
    g = nx.from_numpy_array(A)

    # Add gene names as attributes
    name_mapping = {i: gene_names[i] for i in range(len(gene_names))}
    nx.set_node_attributes(g, name_mapping, 'name')
    print('\tGraph prepareddd')

    return g


def map_pathway_to_nodes():
    '''Returns a dataframe mapping each pathway to the nodes (names & indices) it contains.
    Includes two clusterings apart from the pathways extracted from UniProt:
    (i) "Unmapped": includes all genes with either NaN pathways or belonging to a single-gene pathway
     (ii) "Unknown": includes all genes from CRISPR not belonging to any pathway or to "unmapped".
    '''
    # Load gene names (extracted from CRISPR) and map names to the vertices' indices
    genes = pd.read_csv('../datasets/names.txt', header=None)[0]
    gene_to_index = {name: idx for idx, name in enumerate(genes)}

    # Load the data downloaded from UniProt
    dfs = []
    for i in range(4):
        df = pd.read_excel(f'../datasets/uniprot/uniprot{i + 1}.xlsx')
        dfs.append(df)
    df = pd.concat([i for i in dfs], ignore_index=True)
    df = df[df['Reviewed'] == 'reviewed'].reset_index().iloc[:, 1:] # Only take reviewed samples (all but 11 have only one reviewed)
    df = df[['From', 'Reactome']]   # Select columns with gene name and pathway name

    # Remove nan values and add them to unmapped_names
    unmapped = df[df['Reactome'].isna()]
    unmapped_list = list(np.unique(unmapped['From'].values))
    df = df.dropna(subset=['Reactome']).reset_index().iloc[:, 1:]

    # Create an extended dataframe with a 1 to 1 mapping from every gene to all the pathways it belongs to
    name_list, pathway_list = [], []
    for idx, row in df.iterrows():
        pathways = str(df.iloc[idx, 1])
        pathways = [pathway for pathway in pathways.split(';') if pathway != '']
        for pathway in pathways:
            name_list.append(df.iloc[idx, 0])
            pathway_list.append(pathway)
    extended = pd.DataFrame({'Genes': name_list, 'Pathway': pathway_list})

    # Group by pathway
    pathway_to_genes = extended.groupby('Pathway')['Genes'].agg(list).reset_index()
    # We'll take single-gene pathways to be in the same classification as those genes without a pathway
    unmapped = pathway_to_genes[[len(genes) < 2 for genes in pathway_to_genes['Genes'].values]]
    unmapped_list.extend([gene[0] for gene in np.unique(unmapped['Genes'].values)])
    # Classify the rest of the CRISPR genes as unknown
    unknown_list = [gene for gene in genes if (not gene in extended['Genes'].values)
                                            & (not gene in unmapped_list)]
    pathway_to_genes = pathway_to_genes[[len(genes) > 1 for genes in pathway_to_genes['Genes'].values]].reset_index().iloc[:, 1:]
    # Add "unmapped" and "unknown" to the DF
    pathway_to_genes.loc[len(pathway_to_genes)] = pd.Series({'Pathway': 'No pathway', 'Genes': unmapped_list})
    pathway_to_genes.loc[len(pathway_to_genes)] = pd.Series({'Pathway': 'Unknown pathway', 'Genes': unknown_list})

    # Add node indices
    nodes = []
    for idx, row in pathway_to_genes.iterrows():
        pathway_nodes = [gene_to_index[gene] for gene in row['Genes'] if gene in gene_to_index]
        nodes.append(pathway_nodes)
    pathway_to_genes['Nodes'] = nodes

    # Remove duplicate pathways consisting of a set of nodes already present in the DF
    df = pathway_to_genes.drop_duplicates(subset='Nodes', keep='first').reset_index().iloc[:, 1:]
    df['# genes'] = df['Nodes'].apply(len)

    return df


def get_cliques_from_pathway(g, pathway_to_nodes, pathway_idx, progress_list):
    gt.openmp_set_num_threads(15)

    # Create subgraph of the specific pathway
    pathway_nodes = set(pathway_to_nodes.iloc[pathway_idx, 2])  # Node indices of the genes in the pathway
    pathway_name = str(pathway_to_nodes.iloc[pathway_idx, 0])   # Pathway name
    mask = g.new_vertex_property('bool', vals=[int(v) in pathway_nodes for v in g.vertices()])
    subgraph = gt.GraphView(g, vfilt=mask)  # Subgraph
    ne = subgraph.num_edges()
    nn = subgraph.num_vertices()
    del pathway_to_nodes, pathway_nodes, mask

    min_entropy = float('inf')
    best_state = None

    # Initialise clique state
    if subgraph.num_edges() == 0:
        hyperedges = None
        best_state = 1
        state = 1
        time_cliques = None
        time_sweeps = None
    else:
        # Initialise maximal cliques
        ts1 = time.time()
        state = CliqueState(subgraph)
        ts2 = time.time()
        time_cliques = ts2 - ts1

        # Perform Metropolis-Hastings MCMC
        tm1 = time.time()

        # With burn-in period (no inv. T)
        state.mcmc_sweep(niter=10000, beta=1)   # Default inverse temperature
        for i in range(1000):
            state.mcmc_sweep(niter=1, beta=1)
            current_entropy = state.entropy()
            if current_entropy < min_entropy:
                min_entropy = current_entropy
                best_state = state.copy()

        hyperedges = []
        for v in best_state.f.vertices():  # iterate through factor graph
            if best_state.is_fac[v]:
                continue  # skip over factors (pairwise connections)
            #print(best_state.c[v], best_state.x[v])  # clique occupation
            if best_state.x[v] > 0:
                hyperedges.append(best_state.c[v])

        hyperedges = [tuple(hyperedge) for hyperedge in hyperedges]
        tm2 = time.time()
        print(f'\n\thyperedges of {nn} nodes: {round(tm2 - tm1, 3)} s')
        time_sweeps = tm2 - tm1
    del subgraph, state, best_state

    progress_list.append(1)
    return (pathway_name, hyperedges, ne, nn, time_cliques, time_sweeps, min_entropy)


def pathway_clustering_specific(g):
    gene_names = list(pd.read_csv('../datasets/names.txt', header=None)[0].values)
    index_to_name = {idx: name for idx, name in enumerate(gene_names)}

    pathway_to_nodes = map_pathway_to_nodes()
    pathway_to_nodes = pathway_to_nodes.sort_values(by=['# genes']).reset_index().iloc[:, 1:]

    with Manager() as manager:
        progress_list = manager.list()
        n = 1783
        result = get_cliques_from_pathway(g, pathway_to_nodes, 1783, progress_list)

        name = result[0]
        indices = result[1]
        num_he = result[2]
        num_no = result[3]
        time_c = result[4]
        time_s = result[5]
        min_entr = result[6]


    df = pd.DataFrame({'Pathway': name,
                       'Hyperedges (vertex indices)': indices,
                       '# Nodes': num_no,
                       '# Hyperedges': num_he,
                       'Time cliques': time_c,
                       'Time sweeps': time_s,
                       'Entropy': min_entr})
    print('Doing OK...')

    pathway_genes = []
    for hyperedge in indices:
        hyperedge_genes = [index_to_name[node] for node in hyperedge]
        pathway_genes.append(hyperedge_genes)

    df['Hyperedges (gene names)'] = pathway_genes

    return df


def pathway_clustering(g):
    gene_names = list(pd.read_csv('../datasets/names.txt', header=None)[0].values)
    index_to_name = {idx: name for idx, name in enumerate(gene_names)}

    pathway_to_nodes = map_pathway_to_nodes()
    pathway_to_nodes = pathway_to_nodes.sort_values(by=['# genes']).reset_index().iloc[:, 1:]

    with Manager() as manager:
        progress_list = manager.list()
        n = 1783
        with Pool(processes=1) as pool:
            results = [pool.apply_async(get_cliques_from_pathway, args=(g, pathway_to_nodes, pathway_idx, progress_list))
                       for pathway_idx in range(n)]
            pool.close()
            with tqdm(total=n) as pbar:
                while len(progress_list) < n:
                    pbar.update(len(progress_list) - pbar.n)
                    pbar.refresh()
                pbar.update(n - pbar.n)
            pool.join()

        pathway_names = [result.get()[0] for result in results]
        hyperedges_indices = [result.get()[1] for result in results]
        num_hyperedges = [result.get()[2] for result in results]
        num_nodes = [result.get()[3] for result in results]
        time_cliques = [result.get()[4] for result in results]
        time_sweeps = [result.get()[5] for result in results]
        min_entropies = [result.get()[6] for result in results]

    df = pd.DataFrame({'Pathway': pathway_names,
                       'Hyperedges (vertex indices)': hyperedges_indices,
                       '# Nodes': num_nodes,
                       '# Hyperedges': num_hyperedges,
                       'Time cliques': time_cliques,
                       'Time sweeps': time_sweeps,
                       'Entropy': min_entropies})

    hyperedges_genes = []
    for pathway in tqdm(hyperedges_indices):
        pathway_genes = []
        if pathway is None:
            hyperedges_genes.append(np.nan)
            continue
        for hyperedge in pathway:
            hyperedge_genes = [index_to_name[node] for node in hyperedge]
            pathway_genes.append(hyperedge_genes)
        hyperedges_genes.append(pathway_genes)

    pathway_genes = []
    for hyperedge in hyperedges_indices:
        hyperedge_genes = [index_to_name[node] for node in hyperedge]
        pathway_genes.append(hyperedge_genes)

    df['Hyperedges (gene names)'] = pathway_genes

    return df


def create_subgraphs(g, mptn):
    '''
    Returns dictionary mapping pathway name to subgraph object
    g: full CRISPR gene graph
    mptn: dataframe returned by map_pathway_to_nodes()
    '''
    subgraphs = {}
    print('Creating subgraphs')
    for idx, row in tqdm(mptn.iterrows()):
        pathway = row['Pathway']
        nodes = row['Nodes']
        subgraph = g.subgraph(nodes)
        if (subgraph.number_of_edges() > 0) & (subgraph.number_of_nodes() > 2):
            subgraphs[pathway] = g.subgraph(nodes)
        else:
            pass

    return subgraphs


def load_cluster(path):
    nodes = pd.read_csv(path, header=None).T[0].values
    cluster = g.subgraph(nodes)
    return cluster


def spectral_clustering(g):
    mptn = map_pathway_to_nodes()
    subgraphs = create_subgraphs(g, mptn)
    bigbro = subgraphs['No pathway']


    components = list(nx.connected_components(bigbro))
    # Largest connected component
    largest_cc = max(components, key=len)
    g_largest_cc = bigbro.subgraph(largest_cc).copy()
    print(f'Largest component has {g_largest_cc.number_of_nodes()} nodes')
    A_largest_cc = nx.adjacency_matrix(g_largest_cc).toarray()
    eigvals_20 = eigvals(A_largest_cc, k=20)

    plt.rc('text', usetex=True)
    plt.rc('font', family='serif')
    plt.rcParams.update({'font.size': 21})

    fig, ax = plt.subplots(1, 1, figsize=(9, 4))
    ax.plot(np.arange(1, 21), eigvals_20, color='blue', alpha=0.4, linestyle='-', marker='o', markersize=3)
    ax.set_xlabel('Index', fontsize=21)
    ax.set_ylabel('Eigenvalue', fontsize=21)
    ax.set_title(f'Smallest 20 eigenvalues of the Unmapped subgraph with $|V| = 8069$', fontsize=22)
    fig.savefig('picture_eigvals.png', bbox_inches='tight')
    plt.close(fig)

    # Smallest connected component
    smallest_cc = min(components, key=len)
    g_smallest_cc = bigbro.subgraph(smallest_cc).copy()
    print(f'Smallest component has {g_smallest_cc.number_of_nodes()} node')

    print('Spectral clustering the large one')
    sc = SpectralClustering(n_clusters=3, affinity='precomputed')
    labels = sc.fit_predict(A_largest_cc)
    print('\t Doneee')
    return labels, mptn, g_largest_cc


def eigvals(A, k=20):
    '''
    Returns the k least-dominant eigenvalues of matrix A
    '''
    degrees = np.array(A.sum(axis=1)).flatten()
    D = scipy.sparse.diags(degrees)
    L = D - A
    eigvals, eigvecs = scipy.sparse.linalg.eigsh(L, k=k, which='SM')
    return eigvals


def spectral_clustering_load_labels(g):
    # labels = pd.read_csv('spectral_5clusters_bigbro_labels.csv', header=None)[0].values
    # Separate disconnected groups and cluster the second one
    labels, mptn, g_large = spectral_clustering(g)
    print('Labelling nodes')

    for i, node in enumerate(g_large.nodes()):
        g_large.nodes[node]['label'] = labels[i]

    clusters = {}
    unique_labels = np.unique(labels)
    print('Creating clusters...')
    for label in tqdm(unique_labels):
        # Find nodes with this label
        nodes_in_cluster = [node for node, data in g_large.nodes(data=True) if data['label'] == label]
        print(f'Cluster with {len(nodes_in_cluster)} nodes')
        subgraph = g_large.subgraph(nodes_in_cluster).copy()
        clusters[label] = subgraph
    print('Clusters created, returning them')
    return clusters, mptn


def remove_edges_not_in_triangle(g):
    graph = g.copy()
    edges_to_remove = []
    for e in tqdm(graph.edges()):
        u, v = e.source(), e.target()
        neighbors_u = set(graph.iter_all_neighbors(u))
        neighbors_v = set(graph.iter_all_neighbors(v))
        common = neighbors_u & neighbors_v
        if len(common) == 0:
            edges_to_remove.append(e)
    for e in edges_to_remove:
        graph.remove_edge(e)
    return graph


def remove_non_triangle_edges(g):
    """
    Removes edges from the graph that do not form part of a triangle.

    Parameters:
    g (Graph or GraphView): The input graph or graph view.
    """
    # Create a set to keep edges that are part of a triangle
    triangle_edges = set()

    # Iterate over all vertices in the graph
    for v in tqdm(g.vertices()):
        # Find the neighbors of the vertex
        neighbors = set(v.out_neighbors())

        # Iterate over pairs of neighbors
        for u in neighbors:
            for w in neighbors:
                if u != w and g.edge(u, w) is not None:
                    # If an edge exists between u and w, it forms a triangle with v
                    triangle_edges.add(g.edge(v, u))
                    triangle_edges.add(g.edge(v, w))
                    triangle_edges.add(g.edge(u, w))

    # Collect edges to be removed
    edges_to_remove = [e for e in g.edges() if e not in triangle_edges]

    # Remove the edges from the graph
    for e in edges_to_remove:
        g.remove_edge(e)

    return None


def nx_to_gt(graph):
    gt_graph = gt.Graph(directed=False)
    gt_graph.vertex_properties['node_label'] = gt_graph.new_vertex_property('int')
    node_map = {}

    for node in graph.nodes():
        v = gt_graph.add_vertex()
        node_map[node] = v
        gt_graph.vp.node_label[v] = node

    for edge in graph.edges(data=True):
        u, v, data = edge
        gt_graph.add_edge(node_map[u], node_map[v])

    return gt_graph


def edge_density(g):
    '''Returns global edge density of a Graph instance'''
    return g.num_edges() / (g.num_vertices() * (g.num_vertices() - 1) / 2)


def extractfromrow(row):
    return set(eval(row['Hyperedges']))


def jaccard_similarity(set1, set2):
    intersection = set1.intersection(set2)
    union = set1.union(set2)
    return len(intersection) / len(union)


def compare_hyperedges(dfs):
    n_rows = len(dfs[0])
    results = []
    for i in range(n_rows):
        row_sets = [extractfromrow(df.iloc[i]) for df in dfs]
        row_results = []
        for j in range(len(row_sets)):
            for k in range(j+1, len(row_sets)):
                similarity = jaccard_similarity(row_sets[j], row_sets[k])
                row_results.append(similarity)
        results.append(row_results)
    return pd.DataFrame(results)


if __name__ == '__main__':
    # time1 = time.time()
    # g = prep_graph(0.2, False)
    # df = pathway_clustering(g)
    # print('Saving data frame...')
    # df.to_csv('hyperedges/bigone.csv', index=None)
    # time2 = time.time()
    # print(time2-time1)

    g = prep_graph_networkx(0.2, False)
    spectral_clustering(g)
