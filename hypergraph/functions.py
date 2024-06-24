import graph_tool.all as gt
from graph_tool.inference import CliqueState
import pandas as pd
import numpy as np
import time
from tqdm import tqdm
import matplotlib.pyplot as plt
from multiprocessing import Pool, Manager


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

    # Initialise clique state
    if subgraph.num_edges() == 0:
        cliques = None
        state = 1
    else:
        print('clique state...')
        # Initialise maximal cliques
        ts1 = time.time()
        state = CliqueState(subgraph)
        ts2 = time.time()
        print(f'\tMaximal cliques done in {round(ts2-ts1, 1)} s')
        # Perform Metropolis-Hastings MCMC initial sweep to ensure convergence to the real distribution
        tm1 = time.time()
        state.mcmc_sweep(niter=10000)
        tm2 = time.time()
        print(f'\titerations done in {round(tm2-tm1, 1)} s')
        cliques = []
        for v in state.f.vertices():  # iterate through factor graph
            if state.is_fac[v]:
                continue  # skip over factors
            #print(state.c[v], state.x[v])  # clique occupation
            if state.x[v] > 0:
                cliques.append(state.c[v])

        cliques = [list(clique) for clique in cliques]
    del subgraph, state

    progress_list.append(1)
    return (pathway_name, cliques, ne, nn)

#get most probable hyperedges
def save_pathway_subgraph(pathway, subgraph, progress_list):
    f = f'pathways/pathway_{pathway}.gt'
    subgraph.save(f)
    progress_list.append(1)


def load_pathway_subgraph(index):
    subgraph = gt.Graph()
    subgraph.load(f'pathways/pathway_{index}.gt')
    return subgraph


def pathway_clustering(g):
    '''
    DON'T MIND WHAT'S ALREADY WRITTEN:
        THE PURPOSE OF THIS FUNCTION IS TO LOOP OVER EVERY PATHWAY IN MAP_PATHWAY_TO_NODES(), OBTAIN ITS CLIQUES
        USING GET_CLIQUES_FROM_PATHWAY() AND SAVE ALL IN A DATAFRAME WITH COLUMNS ['PATHWAY', 'CLIQUES'].
        PREFERRABLY, THE 'CLIQUES' COLUMN CONTAINS GENE NAMES INSTEAD OF INDICES
    '''
    gene_names = list(pd.read_csv('../datasets/names.txt', header=None)[0].values)
    index_to_name = {idx: name for idx, name in enumerate(gene_names)}

    pathway_to_nodes = map_pathway_to_nodes()

    with Manager() as manager:
        progress_list = manager.list()
        n = 472
        with Pool(processes=15) as pool:
            #results = [pool.apply_async(get_cliques_from_pathway, args=(g, pathway_to_nodes, pathway_idx, progress_list))
                       #for pathway_idx in range(len(pathway_to_nodes))]
            results = [pool.apply_async(get_cliques_from_pathway, args=(g, pathway_to_nodes, pathway_idx, progress_list))
                       for pathway_idx in range(472)]
            pool.close()
            with tqdm(total=n) as pbar:
                while len(progress_list) < n:
                    pbar.update(len(progress_list) - pbar.n)
                    pbar.refresh()
                pbar.update(n - pbar.n)
            #pool.join()

            pathway_names = [result.get()[0] for result in results]
            cliques_indices = [result.get()[1] for result in results]
    print('Doing OK...')

    df = pd.DataFrame({'Pathway': pathway_names, 'Cliques (vertex indices)': cliques_indices})

    cliques_genes = []
    for pathway in tqdm(cliques_indices):
        pathway_genes = []
        if pathway is None:
            cliques_genes.append(np.nan)
            continue
        for clique in pathway:
            clique_genes = [index_to_name[idx] for idx in clique]
            pathway_genes.append(clique_genes)
        cliques_genes.append(pathway_genes)

    df['Cliques (gene names)'] = cliques_genes

    return df


def pathway_clustering2(g):
    gene_names = list(pd.read_csv('../datasets/names.txt', header=None)[0].values)
    index_to_name = {idx: name for idx, name in enumerate(gene_names)}

    pathway_to_nodes = map_pathway_to_nodes()

    n = 400

    pl = []
    pathway_names = []
    cliques_indices = []
    for i in tqdm(range(n)):
        results = get_cliques_from_pathway(g, pathway_to_nodes, i, pl)
        pathway_names.append(results[0])
        cliques_indices.append(results[1])

    df = pd.DataFrame({'Pathway': pathway_names, 'Cliques (vertex indices)': cliques_indices})

    cliques_genes = []
    for pathway in cliques_indices:
        pathway_genes = []
        if pathway is None:
            cliques_genes.append(np.nan)
            continue
        for clique in pathway:
            clique_genes = [index_to_name[idx] for idx in clique]
            pathway_genes.append(clique_genes)
        cliques_genes.append(pathway_genes)

    df['Cliques (gene names)'] = cliques_genes

    return df


def edge_density(g):
    '''Returns global edge density of a Graph instance'''
    return g.num_edges() / (g.num_vertices() * (g.num_vertices() - 1) / 2)


if __name__ == '__main__':
    g = prep_graph(0.9, True)
    df = pathway_clustering(g)
    print('Saving data frame...')
    df.to_csv('cliques_trial2.csv', index=None)
