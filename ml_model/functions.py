import os
import csv
import pandas as pd
import numpy as np
import re
import scipy.sparse
import scipy.sparse.linalg
from collections import defaultdict
from sklearn.cluster import SpectralClustering
from node2vec import Node2Vec
from tqdm import tqdm
import networkx as nx
from networkx.algorithms.centrality import betweenness_centrality, closeness_centrality
import hypergraphx as hx
from hypergraphx.measures.eigen_centralities import power_method
from concurrent.futures import ThreadPoolExecutor, as_completed


def clean_col_names(col):
    '''Takes a label from the CRISPR dataset which have a structure of 'GeneName (GeneID)'
    and returns only 'GeneName' '''
    return col.split(' (')[0]


def prep_graph(threshold, ranked):
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
    print('loco estamos en ello')
    # Create adjacency matrix A
    A = (corrs.values > threshold).astype(np.int8)
    # Create NetworkX graph object from A
    g = nx.from_numpy_array(A)

    # Add gene names as attributes
    name_mapping = {i: gene_names[i] for i in range(len(gene_names))}
    nx.set_node_attributes(g, name_mapping, 'name')

    return g


def prep_hypergraphs(mptn):
    '''Returns dictionary mapping pathway names to hypergraphx hypergraph'''
    # Load the lists of hyperedges for every pathway
    hyperedges_df = pd.read_csv('../hypergraph/hyperedges/hyperedges_final.csv')
    hypergraphs = {}

    # Create hypergraph objects
    for idx, row in hyperedges_df.iterrows():
        pathway = row['Pathway']
        list_nodes = mptn.loc[mptn['Pathway'] == pathway]['Nodes'].values[0]
        h = hx.Hypergraph()
        h.add_nodes(list_nodes)
        list_hyperedges = eval(row['Hyperedges'])
        h.add_edges(list_hyperedges)
        hypergraphs[row['Pathway']] = h

    return hypergraphs


def map_pathway_to_nodes():
    '''Returns a dataframe mapping each pathway to the nodes (names & indices) it contains.
    Includes two clusterings apart from the pathways extracted from UniProt:
    (i) "Unmapped": includes all genes with either NaN pathways or belonging to a single-gene pathway
     (ii) "Unknown": includes all genes from CRISPR not belonging to any pathway or to "unmapped".
    '''
    # Load gene names (extracted from CRISPR) and map names to the vertices' indices
    genes = pd.read_csv('../datasets/names.txt', header=None)[0]
    gene_to_index = {name: idx for idx, name in enumerate(genes)}
    index_to_gene = {idx: gene for idx, gene in enumerate(genes)}

    # Load the data downloaded from UniProt
    dfs = []
    for i in range(4):
        df = pd.read_excel(f'../datasets/uniprot/uniprot{i + 1}.xlsx')
        dfs.append(df)
        print('lol?')
    df = pd.concat([i for i in dfs], ignore_index=True)
    print('okay')
    df = df[df['Reviewed'] == 'reviewed'].reset_index().iloc[:, 1:] # Only take reviewed samples (all but 11 have only one reviewed)
    print('notokay?')
    df = df[['From', 'Reactome']]   # Select columns with gene name and pathway name

    # Remove nan values and add them to unmapped_names
    unmapped = df[df['Reactome'].isna()]
    unmapped_list = list(np.unique(unmapped['From'].values))
    df = df.dropna(subset=['Reactome']).reset_index().iloc[:, 1:]
    print('going')

    # Create an extended dataframe with a 1 to 1 mapping from every gene to all the pathways it belongs to
    name_list, pathway_list = [], []
    for idx, row in df.iterrows():
        pathways = str(df.iloc[idx, 1])
        pathways = [pathway for pathway in pathways.split(';') if pathway != '']
        for pathway in pathways:
            name_list.append(df.iloc[idx, 0])
            pathway_list.append(pathway)
    extended = pd.DataFrame({'Genes': name_list, 'Pathway': pathway_list})
    print('yeaaa')

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
    print('almost there')

    # Remove duplicate pathways consisting of a set of nodes already present in the DF
    df = pathway_to_genes.drop_duplicates(subset='Nodes', keep='first').reset_index().iloc[:, 1:]
    df['# genes'] = df['Nodes'].apply(len)

    # Add unmapped clusters
    # Small cluster
    small_nodes = np.sort(pd.read_csv('../hypergraph/hyperedges/unmapped_smallcluster_nodes.csv', header=None).T[0].values)
    small_genes = [index_to_gene[idx] for idx in small_nodes]
    df.loc[len(df)] = ['U-small', small_genes, small_nodes, len(small_nodes)]

    # Medium cluster
    for i in range(17):
        mid_nodes = np.sort(pd.read_csv(f'../hypergraph/hyperedges/unmapped_middlecluster_nodes_{i}.csv', header=None).T[0].values)
        mid_genes = [index_to_gene[idx] for idx in mid_nodes]
        df.loc[len(df)] = [f'U-mid-{i}', mid_genes, mid_nodes, len(mid_nodes)]

    # Big cluster
    big_small_nodes = np.sort(pd.read_csv('../hypergraph/hyperedges/unmapped_bigcluster_smallsubcluster_nodes.csv', header=None).T[0].values)
    big_small_genes = [index_to_gene[idx] for idx in big_small_nodes]
    df.loc[len(df)] = ['U-big-small', big_small_genes, big_small_nodes, len(big_small_nodes)]

    for i in range(7):
        big_mid_nodes = np.sort(pd.read_csv(f'../hypergraph/hyperedges/unmapped_bigcluster_midsubcluster_{i}.csv', header=None).T[0].values)
        big_mid_genes = [index_to_gene[idx] for idx in big_mid_nodes]
        df.loc[len(df)] = [f'U-big-mid-{i}', big_mid_genes, big_mid_nodes, len(big_mid_nodes)]

    for i in range(8):
        if i != 2:
            big_big_nodes = np.sort(pd.read_csv(f'../hypergraph/hyperedges/unmapped_bigcluster_bigsubcluster_nodes_{i}.csv', header=None).T[0].values)
            big_big_genes = [index_to_gene[idx] for idx in big_big_nodes]
            df.loc[len(df)] = [f'U-big-big-{i}', big_big_genes, big_big_nodes, len(big_big_nodes)]
        else:
            for j in range(3):
                big_big_nodes = np.sort(pd.read_csv(f'../hypergraph/hyperedges/unmapped_bigcluster_bigsubcluster_nodes_2_{j}.csv', header=None).T[0].values)
                big_big_genes = [index_to_gene[idx] for idx in big_big_nodes]
                df.loc[len(df)] = [f'U-big-big-2.{j}', big_big_genes, big_big_nodes, len(big_big_nodes)]

    df = df.drop(df[df['Pathway'] == 'No pathway'].index)
    print('gogogo')

    return df


def create_subgraphs(g, mptn):
    '''
    Returns dictionary mapping pathway name to subgraph object
    g: full CRISPR gene graph
    mptn: dataframe returned by map_pathway_to_nodes()
    '''
    subgraphs = {}
    nodess = []
    print('Creating subgraphs')
    for idx, row in tqdm(mptn.iterrows()):
        pathway = row['Pathway']
        if pathway == 'No pathway':
            continue
        nodes = row['Nodes']
        subgraph = g.subgraph(nodes)
        nodes_sub = list(subgraph.nodes())

        if (subgraph.number_of_edges() > 0) & (subgraph.number_of_nodes() > 2):
            subgraphs[pathway] = g.subgraph(nodes)
            nodess.append(nodes_sub)
        else:
            pass

    return subgraphs


def obtain_graph_metrics(g, mptn):
    '''
    Computes degree, betweenness, closeness, and eigenvector centralities
    g: graph-tool or networkx object
    mptn: output of map_pathway_to_nodes()
    subgraphs: output of create_subgraphs(g, mptn)
    '''
    subgraphs = create_subgraphs(g, mptn)
    i = 0
    metrics = {}
    for pathway, subgraph in tqdm(subgraphs.items()):
        print(f'Beginning {i}')
        values = {}
        nn = subgraph.number_of_nodes()

        if nn == 2:
            continue

        # Degree centrality
        values['degree'] = nx.degree_centrality(subgraph)
        print('Degree done.')
        # Betweenness centrality
        values['betweenness'] = nx.betweenness_centrality(subgraph)
        print('Betweenness done!!!')
        # Closeness centrality
        values['closeness'] = nx.closeness_centrality(subgraph)
        print('Closeness done')
        # Eigenvector centrality
        values['eigenvector'] = nx.eigenvector_centrality(subgraph, max_iter=5000, tol=1e-6)
        # Clustering coefficient
        values['clustering'] = nx.clustering(subgraph)
        # Bottleneck coefficient
        values['bottleneck'] = bottleneck(subgraph)

        metrics[pathway] = values
        print(i)
        i += 1

    data = []
    for pathway, metrics in tqdm(metrics.items()):
        for node in metrics['degree']:
            data.append({
                'Gene': node,
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


def obtain_hypergraph_metrics(hypergraphs, weird=True):
    i = 0
    metrics = {}
    for pathway, h in tqdm(hypergraphs.items()):
        print(f'\nMetrics of {i}, with {h.num_nodes()} nodes and {h.num_edges()} hyperedges')
        values = {}

        # Degree sequence
        deg_seq = h.degree_sequence()

        # Compute adjacency matrix of star expansion (bipartite graph)
        inc, mapping = h.incidence_matrix(return_mapping=True)
        inc_dense = inc.toarray()
        n_nodes = h.num_nodes()
        n_edges = h.num_edges()
        zeros_nodes = np.zeros((n_nodes, n_nodes))
        zeros_edges = np.zeros((n_edges, n_edges))
        A = np.vstack((np.hstack((zeros_nodes, inc_dense)), np.hstack((inc_dense.T, zeros_edges))))
        star_expansion = nx.from_numpy_array(A)

        # Scale degree
        values['degree'] = {k: v / (n_nodes - 1) for k, v in deg_seq.items()}

        # Betweenness
        if weird:
            betweenness = {idx: 0.0 for idx, node in enumerate(star_expansion.nodes()) if idx < n_nodes}
            hg_nodes = [idx for idx, node in enumerate(star_expansion.nodes()) if idx < n_nodes]
            print(f'Number of sources: {len(hg_nodes)}')
            for j, s in enumerate(tqdm(hg_nodes)):
                for t in hg_nodes[j+1:]:
                    try:
                        all_shortest_paths = list(nx.all_shortest_paths(star_expansion, source=s, target=t))
                    except nx.exception.NetworkXNoPath:
                        continue
                    total_paths = len(all_shortest_paths)

                    for path in all_shortest_paths:
                        for node in path:
                            if node != s and node != t and node in hg_nodes:
                                betweenness[node] += 1 / total_paths
            if n_nodes > 2:
                scale = 1 / ((n_nodes - 1) * (n_nodes - 2) / 2)
            else:
                scale = 1
            betweenness = {k: v * scale for k, v in betweenness.items()}
            betweenness = {mapping[k]: betweenness[k] for k in range(n_nodes)}
        else:
            betweenness = betweenness_centrality(star_expansion)
            betweenness = {mapping[k]: betweenness[k] for k in range(n_nodes)}
        values['betweenness'] = betweenness
        print('betweenness done')

        # Closeness
        if weird:
            closeness = {node: 0.0 for idx, node in enumerate(star_expansion.nodes()) if idx < n_nodes}
            hg_nodes = [node for idx, node in enumerate(star_expansion.nodes()) if idx < n_nodes]
            print(f'Number of sources: {len(hg_nodes)}')

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
        else:
            closeness = closeness_centrality(star_expansion)
            closeness = {mapping[k]: closeness[k] for k in range(n_nodes)}
        values['closeness'] = closeness
        print('closeness done')

        # Eigencentrality
        print('Starting power method')
        eigencentrality = power_method(A, max_iter=10000, tol=1e-8)
        values['eigencentrality'] = {mapping[k]: eigencentrality[k] for k in range(n_nodes)}
        print('eigenpollas hecho')

        # Clustering
        n_neighbours = {node: 2**(len(h.get_neighbors(node))+1) - len(h.get_neighbors(node)) - 2 for node in h.get_nodes()}
        n_hyperedges = {node: len(h.get_incident_edges(node)) for node in h.get_nodes()}
        clustering_trial = {}
        for node in h.get_nodes():
            if n_neighbours[node] != 0:
                clustering_trial[node] = n_hyperedges[node] / n_neighbours[node]
            else:
                clustering_trial[node] = 0

        clustering = {idx: clustering_trial[node] for idx, node in enumerate(h.get_nodes())}
        print('clustering fokin done')

        clustering = {mapping[k]: clustering[k] for k in range(n_nodes)}
        values['clustering'] = clustering

        # Bottleneck
        print('bottlepolla')
        bn = bottleneck_hypergraph(star_expansion, n_nodes)
        bn = {mapping[k]: bn[k] for k in range(n_nodes)}
        values['bn'] = bn

        metrics[pathway] = values
        print(i)
        i += 1

    data = []
    for pathway, measures in tqdm(metrics.items()):
        for node in measures['degree']:
            try:
                data.append({
                    'Gene': node,
                    'Pathway': pathway,
                    'Degree': measures['degree'][node],
                    'Betweenness': measures['betweenness'][node],
                    'Closeness': measures['closeness'][node],
                    'Eigencentrality': measures['eigencentrality'][node],
                    'Clustering coef.': measures['clustering'][node],
                    'Bottleneck': measures['bn'][node]
                })
            except KeyError:
                print('brooo')

    metrics_df = pd.DataFrame(data)
    metrics_df = metrics_df.sort_values(by=['Gene', 'Pathway'], ascending=True)
    return metrics_df


def obtain_graph_embeddings(g, mptn, dim=20, walk_length=30, num_walks=200, workers=4):
    subgraphs = create_subgraphs(g, mptn)

    names = pd.read_csv('../datasets/names.txt', header=None)[0]
    idx_to_name = {idx: name for idx, name in enumerate(names)}

    i = 0
    dfs = []
    for pathway, subgraph in tqdm(subgraphs.items()):
        print(f'Embedding {i} with {subgraph.number_of_nodes()} nodes and {subgraph.number_of_edges()} edges')

        node2vec = Node2Vec(subgraph, dimensions=dim, walk_length=walk_length, num_walks=num_walks, workers=workers)
        print('Node2Vec created, now fitting')
        model = node2vec.fit(window=10, min_count=1, batch_words=4)
        print('Fitted?')
        embeddings = model.wv

        embedding_df = pd.DataFrame([embeddings[str(node)] for node in subgraph.nodes()])
        embedding_df['Gene'] = list(subgraph.nodes())
        embedding_df['Pathway'] = [pathway for _ in range(len(embedding_df))]
        dfs.append(embedding_df)
        i += 1

    df = pd.concat(dfs, ignore_index=True)

    df_names = [idx_to_name[idx] for idx in df['Gene']]
    df['Gene name'] = df_names

    cols_to_move = ['Gene', 'Gene name', 'Pathway']
    new_order = cols_to_move + [col for col in df.columns if col not in cols_to_move]
    df = df[new_order]

    df = df.sort_values(by=['Gene', 'Pathway']).reset_index().iloc[:, 1:]

    return df


def obtain_hypergraph_embeddings(hypergraphs, nodes_vs_hyperedges, dim=20, walk_length=60, num_walks=200, workers=4):

    names = pd.read_csv('../datasets/names.txt', header=None)[0]
    idx_to_name = {idx: name for idx, name in enumerate(names)}

    i = 0
    dfs = []
    for pathway, h in tqdm(hypergraphs.items()):
        print(f'Embedding {i} with {h.num_nodes()} nodes and {h.num_edges()} hyperedges')

        inc, mapping = h.incidence_matrix(return_mapping=True)
        inc_dense = inc.toarray()
        n_nodes = h.num_nodes()
        n_hyperedges = h.num_edges()
        zeros_nodes = np.zeros((n_nodes, n_nodes))
        zeros_edges = np.zeros((n_hyperedges, n_hyperedges))
        A = np.vstack((np.hstack((zeros_nodes, inc_dense)), np.hstack((inc_dense.T, zeros_edges))))
        star_expansion = nx.from_numpy_array(A)

        node2vec = Node2Vec(star_expansion, dimensions=dim, walk_length=walk_length, num_walks=num_walks, workers=workers)
        print('Node2Vec created, now fitting')
        model = node2vec.fit(window=10, min_count=1, batch_words=4)
        print('Fitted?')
        embeddings = model.wv

        if nodes_vs_hyperedges == 'nodes':
            embedding_nodes = pd.DataFrame([embeddings[str(node)] for idx, node in enumerate(star_expansion.nodes()) if idx < n_nodes])
            embedding_nodes['Gene'] = h.get_nodes()
            embedding_nodes['Pathway'] = [pathway for _ in range(len(embedding_nodes))]
            dfs.append(embedding_nodes)
        elif nodes_vs_hyperedges == 'hyperedges':
            embedding_he = pd.DataFrame([embeddings[str(node)] for idx, node in enumerate(star_expansion.nodes()) if idx >= n_nodes])
            #embedding_he['']
        i += 1

    df = pd.concat(dfs, ignore_index=True)

    df_names = [idx_to_name[idx] for idx in df['Gene']]
    df['Gene name'] = df_names

    cols_to_move = ['Gene', 'Gene name', 'Pathway']
    new_order = cols_to_move + [col for col in df.columns if col not in cols_to_move]
    df = df[new_order]

    df = df.sort_values(by=['Gene', 'Pathway']).reset_index().iloc[:, 1:]

    return df


def obtain_hypergraph_metrics2(hypergraphs):
    np.random.seed(143)

    i = 0
    metrics = {}
    for pathway, h in tqdm(hypergraphs.items()):
        print(f'\nMetrics of {i}, with {h.num_nodes()} nodes and {h.num_edges()} hyperedges')
        values = {}

        # Degree sequence
        deg_seq = h.degree_sequence()

        # Compute adjacency matrix of star expansion (bipartite graph)
        inc, mapping = h.incidence_matrix(return_mapping=True)
        inc_dense = inc.toarray()
        n_nodes = h.num_nodes()
        n_edges = h.num_edges()
        zeros_nodes = np.zeros((n_nodes, n_nodes))
        zeros_edges = np.zeros((n_edges, n_edges))
        A = np.vstack((np.hstack((zeros_nodes, inc_dense)), np.hstack((inc_dense.T, zeros_edges))))
        star_expansion = nx.from_numpy_array(A)
        hg_nodes = [idx for idx, node in enumerate(star_expansion.nodes()) if idx < n_nodes]

        # Scale degree
        values['degree'] = {k: v / (n_nodes - 1) for k, v in deg_seq.items()}

        # Betweenness
        betweenness = {idx: 0.0 for idx, node in enumerate(star_expansion.nodes()) if idx < n_nodes}
        print(f'Number of sources: {len(hg_nodes)}')

        def calculate_shortest_paths(s, t, hg_nodes):
            result = []
            try:
                all_shortest_paths = list(nx.all_shortest_paths(star_expansion, source=s, target=t))
            except nx.exception.NetworkXNoPath:
                return result

            total_paths = len(all_shortest_paths)
            for path in all_shortest_paths:
                for node in path:
                    if node != s and node != t and node in hg_nodes:
                        result.append((node, 1 / total_paths))
            return result

        with ThreadPoolExecutor(max_workers=50) as executor:
            futures = []
            for j, s in enumerate(tqdm(hg_nodes)):
                for t in hg_nodes[j + 1:]:
                    futures.append(executor.submit(calculate_shortest_paths, s, t, hg_nodes))

            for future in tqdm(as_completed(futures)):
                result = future.result()
                for node, value in result:
                    betweenness[node] += value
        print('\tScaling betweenness values...')

        if n_nodes > 2:
            scale = 1 / ((n_nodes - 1) * (n_nodes - 2) / 2)
        else:
            scale = 1
        betweenness = {k: v * scale for k, v in betweenness.items()}
        betweenness = {mapping[k]: betweenness[k] for k in range(n_nodes)}

        values['betweenness'] = betweenness
        print('betweenness done')

        # Closeness
        closeness = {node: 0.0 for idx, node in enumerate(star_expansion.nodes()) if idx < n_nodes}
        print(f'Number of sources: {len(hg_nodes)}')

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
        print('closeness done')

        # Eigencentrality
        print('Starting power method')
        eigencentrality = power_method(A, max_iter=10000, tol=1e-8)
        values['eigencentrality'] = {mapping[k]: eigencentrality[k] for k in range(n_nodes)}
        print('eigenpollas hecho')

        # Clustering
        n_neighbours = {node: 2**(len(h.get_neighbors(node))+1) - len(h.get_neighbors(node)) - 2 for node in h.get_nodes()}
        n_hyperedges = {node: len(h.get_incident_edges(node)) for node in h.get_nodes()}
        clustering_trial = {}
        for node in h.get_nodes():
            if n_neighbours[node] != 0:
                clustering_trial[node] = n_hyperedges[node] / n_neighbours[node]
            else:
                clustering_trial[node] = 0

        clustering = {idx: clustering_trial[node] for idx, node in enumerate(h.get_nodes())}
        print('clustering fokin done')

        clustering = {mapping[k]: clustering[k] for k in range(n_nodes)}
        values['clustering'] = clustering

        # Bottleneck
        print('bottlepolla')
        bn = bottleneck_hypergraph(star_expansion, n_nodes)
        bn = {mapping[k]: bn[k] for k in range(n_nodes)}
        values['bn'] = bn

        metrics[pathway] = values
        print(i)
        i += 1

    data = []
    for pathway, measures in tqdm(metrics.items()):
        for node in measures['degree']:
            try:
                data.append({
                    'Gene': node,
                    'Pathway': pathway,
                    'Degree': measures['degree'][node],
                    'Betweenness': measures['betweenness'][node],
                    'Closeness': measures['closeness'][node],
                    'Eigencentrality': measures['eigencentrality'][node],
                    'Clustering coef.': measures['clustering'][node],
                    'Bottleneck': measures['bn'][node]
                })
            except KeyError:
                print('brooo')

    metrics_df = pd.DataFrame(data)
    metrics_df = metrics_df.sort_values(by=['Gene', 'Pathway'], ascending=True)
    return metrics_df


def remove_edges_not_in_triangle(graph):
    edges_to_remove = []
    for e in tqdm(graph.edges()):
        u, v = e.source(), e.target()
        neighbors_u = set(graph.iter_all_neighbors(u))
        neighbors_v = set(graph.iter_all_neighbors(v))
        common = neighbors_u & neighbors_v
        if len(common) == 0:
            edges_to_remove.append(e)
    for e in edges_to_remove:
        graph = graph.remove_edge(e)
    return graph


def bottleneck(graph):
    n_nodes = graph.number_of_nodes()
    bottleneck = {node: 0.0 for idx, node in enumerate(graph.nodes()) if idx < n_nodes}
    hg_nodes = [node for idx, node in enumerate(graph.nodes()) if idx < n_nodes]

    for root in hg_nodes:
        shortest_tree = nx.single_source_shortest_path(graph, root)
        # Count number of paths passing through each node
        path_counts = defaultdict(int)
        for target, path in shortest_tree.items():
            if target == root:
                continue
            for node in path:
                if node != root and node != target and node in hg_nodes:
                    path_counts[node] += 1

        for node, count in path_counts.items():
            if count > n_nodes / 4:
                bottleneck[node] += 1

    return bottleneck


def bottleneck_hypergraph(graph, n_nodes):
    bottleneck = {node: 0.0 for idx, node in enumerate(graph.nodes()) if idx < n_nodes}
    hg_nodes = [node for idx, node in enumerate(graph.nodes()) if idx < n_nodes]

    for root in hg_nodes:
        shortest_tree = nx.single_source_shortest_path(graph, root)
        # Count number of paths passing through each node
        path_counts = defaultdict(int)
        for target, path in shortest_tree.items():
            if target == root:
                continue
            for node in path:
                if node != root and node != target and node in hg_nodes:
                    path_counts[node] += 1

        components = list(nx.connected_components(graph))
        for component in components:
            for node, count in path_counts.items():
                if (count > n_nodes / 4) and (node in component):
                    bottleneck[node] += 1

    return bottleneck


def add_hypergraph(n_nodes, path, pathway_name):
    '''Appends to the dataframe of hypergraphs the hyperedges of a hypergraph
     stored as a .csv file, where each row contains a sequence of nodes, e.g.:
    132, 134, 153, 13
    1423, 13, 2, 145, 14
    2, 13'''
    hyperedges = pd.read_csv('../hypergraph/hyperedges/hyperedges.csv')
    with open(path, 'r') as file:
        reader = csv.reader(file)
        data = [tuple(map(int, row)) for row in reader]

    gene_names = list(pd.read_csv('../datasets/names.txt', header=None)[0].values)
    idx_to_name = {idx: name for idx, name in enumerate(gene_names)}

    genes = []
    for hyperedge in data:
        h_genes = [idx_to_name[node] for node in hyperedge]
        genes.append(h_genes)
    n_hyperedges = len(data)

    hyperedges.loc[len(hyperedges)] = [pathway_name, data, n_nodes, n_hyperedges, 0, 0, 0, genes]
    hyperedges.to_csv('hyperedges.csv', index=None)


def eigvals_nopathway():
    g = prep_graph(0.2, False)
    mptn = map_pathway_to_nodes()
    subgraphs = create_subgraphs(g, mptn)
    subgraph = subgraphs['No pathway']
    A = nx.adjacency_matrix(subgraph).astype(float)
    degrees = np.array(A.sum(axis=1)).flatten()
    D = scipy.sparse.diags(degrees)
    L = D - A
    eigvals, eigvecs = scipy.sparse.linalg.eigsh(L, k=50, which='SM')
    return eigvals


def spectral_clustering(g, mptn):
    labels = pd.read_csv('spectral_5clusters_bigbro_labels.csv', header=None)[0].values
    subgraphs = create_subgraphs(g, mptn)
    bigbro = subgraphs['No pathway']

    for i, node in enumerate(bigbro.nodes()):
        bigbro.nodes[node]['label'] = labels[i]

    clusters = {}
    unique_labels = np.unique(labels)

    for label in unique_labels:
        # Find nodes with this label
        nodes_in_cluster = [node for node, data in bigbro.nodes(data=True) if data['label'] == label]
        subgraph = bigbro.subgraph(nodes_in_cluster).copy()
        clusters[label] = subgraph

    return clusters


def ensure_edge_weights(graph):
    if 'weight' not in graph.edge_properties:
        weight = graph.new_edge_property('double')
        for e in graph.edges():
            weight[e] = 1.0
        graph.edge_properties['weight'] = weight
