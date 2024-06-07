import os
import graph_tool.all as gt
import pandas as pd
import numpy as np
import time
from tqdm import tqdm
import matplotlib.pyplot as plt
from multiprocessing import Pool, Manager


def get_ranked_corrs():
    '''Saves a square symmetric array of the ranked correlations for every pair of genes
    Saves it in 'rank_transf_symm.csv', with row and column labels.'''
    # Function that returns a vector ranked
    def rank_array(arr):
        '''This function takes a vector (a row)
        and returns a vector of the same length with its ranks starting from the smallest value'''

        # Indices that sort the array
        temp = arr.argsort()
        # Ranks
        ranks = temp.argsort() + 1

        # Compute rank of ties
        sorted = np.sort(arr)
        sorted_ranks = np.sort(ranks).astype('float64')
        for i in range(len(arr)):
            start = 0
            # Check if this element is the start of a series of ties
            if i == 0 or sorted[i] != sorted[i - 1]:
                start = i
            # If it is the end of a series of ties
            if i == len(arr) - 1 or sorted[i] != sorted[i + 1]:
                end = i + 1
                avg_rank = np.mean(sorted_ranks[start:end])
                sorted_ranks[start:end] = avg_rank
        return sorted_ranks[np.argsort(temp)]

    # Open and load the dataframe as an array
    df = pd.read_csv('./../CRISPRGeneDependency.csv', delimiter=',')
    depmap = df.iloc[:, 1:]  # Get rid of cell line names

    # Save gene names as np array
    gene_names = depmap.columns.values

    start_corr = time.time()
    print('Calculating correlations...')

    # Get correlation matrix
    corrs_matrix = depmap.corr()  # Pandas doesn't consider NaN values in .corr()

    # Adjust for NaN due to some genes having a St.Dev.=0
    corrs_matrix.fillna(0, inplace=True)
    np.fill_diagonal(corrs_matrix.values, 0)

    end_corr = time.time()
    time_corr = end_corr - start_corr
    print(f'Time to obtain correlations matrix: {int(time_corr / 60)}min {int(time_corr % 60)}s.')

    # Create array to hold the ranks
    rank_transformation = np.zeros_like(corrs_matrix.values)

    # Iterate over each row to rank
    for idx, cell_line in tqdm(enumerate(corrs_matrix.values)):
        rank_transformation[idx, :] = rank_array(cell_line)

    # Symmetrise by taking the largest rank
    for i in tqdm(range(rank_transformation.shape[0])):
        for j in range(i, rank_transformation.shape[1]):
            if rank_transformation[i, j] > rank_transformation[j, i]:
                rank_transformation[j, i] = rank_transformation[i, j]
            elif rank_transformation[i, j] < rank_transformation[j, i]:
                rank_transformation[i, j] = rank_transformation[j, i]

    # Save as Pandas' object to save row and column labels
    matrix = pd.DataFrame(rank_transformation)
    matrix.columns = gene_names
    matrix.index = gene_names

    # Save as .csv file
    matrix.to_csv('ranked_corrs_2.csv', index=True)


def clean_col_names(col):
    '''Takes a label from the CRISPR dataset which have a structure of 'GeneName (GeneID)'
    and returns only 'GeneName' '''
    return col.split(' (')[0]


def get_genes(complex):
    '''Returns a list of different gene names within a complexes
     Example input: 'PRORP;TRMT10C;HSD17B10'
     Example output: ['PRORP', 'TRMT10C', 'HSD17B10']'''
    return complex.split(';')


def filter_CORUM():
    # Load CORUM dataset
    df = pd.read_csv('humanComplexes.txt', delimiter='\t')

    # Select rows containing these substrings in their 'Cell line' value
    substrings = ['T cell line ED40515',
                  'mucosal lymphocytes',
                  'CLL cells',
                  'monocytes',
                  'THP-1 cells',
                  'bone marrow-derived',
                  'monocytes, LPS-induced',
                  'THP1 cells',
                  'human blood serum',
                  'human blood plasma',
                  'plasma',
                  'CSF',
                  'human leukemic T cell JA3 cells',
                  'erythrocytes',
                  'peripheral blood mononuclear cells',
                  'African Americans',
                  'SKW 6.4',
                  'BJAB cells',
                  'Raji cells',
                  'HUT78',
                  'J16',
                  'H9',
                  'U-937',
                  'Jurkat T',
                  'NB4 cells',
                  'U937',
                  'early B-lineage',
                  'T-cell leukemia',
                  'lymphoblasts',
                  'whole blood and lymph',
                  'human neutrophil-differentiating HL-60 cells',
                  'human peripheral blood neutrophils',
                  'human neutrophils from fresh heparinized human peripheral blood',
                  'human peripheral blood',
                  'HCM',
                  'liver-hematopoietic',
                  'cerebral cortex',
                  'human brain',
                  'pancreatic islet',
                  'human hepatocyte carcinoma HepG2 cells',
                  'Neurophils',
                  'H295R adrenocortical',
                  'frontal cortex',
                  'myometrium',
                  'vascular smooth muscle cells',
                  'Dendritic cells',
                  'intestinal epithelial',
                  'Primary dermal fibroblasts',
                  'HK2 proximal',
                  'brain pericytes',
                  'HepG2',
                  'HEK 293 cells, liver',
                  'normal human pancreatic duct epithelial',
                  'pancreatic ductal adenocarcinoma',
                  'OKH cells',
                  'cultured podocytes',
                  'renal glomeruli',
                  'VSMCs',
                  'differentiated HL-60 cells',
                  'SH-SY5Y cells',
                  'frontal and entorhinal cortex',
                  'SHSY-5Y cells',
                  'hippocampal HT22 cells',
                  'primary neurons',
                  'neurons',
                  'renal cortex membranes',
                  'Kidney epithelial cells',
                  'skeletal muscle cells',
                  'Skeletal muscle fibers',
                  'differentiated 3T3-L1',
                  'brain cortex',
                  'cortical and hippocampal areas',
                  'human H4 neuroglioma',
                  'Thalamus',
                  'HISM',
                  'pancreas',
                  'RCC4',
                  'C2C12 myotube',
                  'XXVI muscle',
                  'SH-SY5Y neuroblastoma',
                  'HCC1143',
                  'Hep-2',
                  'PANC-1',
                  'HEK293T cells',
                  'HEK-293 cells',
                  'heart',
                  'epithelium',
                  'kidney',
                  'heart muscle',
                  'central nervous system',
                  'COS-7 cells',
                  'ciliary ganglion',
                  'striated muscle',
                  'PC12',
                  '293FR cells']
    pattern = '|'.join(substrings)

    # Select rows whose 'Cell line' value is exactly one of these
    exact = ['muscle', '293 cells', 'brain', 'HEK 293 cells']

    # Create Boolean mask selecting all the rows described
    partial_mask = df['Cell line'].str.contains(pattern, case=False, na=False)
    exact_mask = df['Cell line'].isin(exact)
    total_mask = partial_mask | exact_mask

    # Obtain filtered dataframe
    complexes_full = df[total_mask]

    # Select specific columns
    complexes = complexes_full[['ComplexID', 'ComplexName', 'Cell line', 'subunits(Gene name)', 'GO description', 'FunCat description']]

    # Sort by 'Cell line'
    complexes = complexes.sort_values(by=['Cell line'])

    # Exclude 'complexes' including only one subunit/ gene
    mask_mono = [len(complexes['subunits(Gene name)'].values[i].split(';')) > 1 for i in range(len(complexes))]
    complexes = complexes.loc[mask_mono]

    # Save as .csv file
    complexes.to_csv('filtered_complexes.csv', index=False)


def init_worker():
    '''Makes sure each of the CPU cores do not return the same pseudorandom results'''
    seed = os.getpid()
    np.random.seed(seed)


def prep_graph(threshold):
    '''Loads the ranked correlation matrix from 'rank_transf_symm_2.csv', normalises it,
    computes the adjacency matrix from the threshold argument and returns a graph-tool Graph object
    together with a list of the gene names (with the gene IDs removed)'''
    gt.openmp_set_num_threads(4)

    print('Preparing graph...')
    start_time = time.time()
    corrs = pd.read_csv('ranked_corrs.csv', delimiter=',', index_col=0)
    corrs /= np.max(corrs.values)   # Normalise
    gene_names = np.array([clean_col_names(col) for col in corrs.columns])  # Remove gene IDs

    # Create adjacency matrix A
    A = (corrs.values > threshold).astype(int)

    # Instantiate graph-tool Graph and add nodes & edges based on A
    g = gt.Graph(directed=False)
    g.add_vertex(n=A.shape[0])
    edges = np.transpose(np.nonzero(np.triu(A, 1))) # Use k=1 to prevent self-interactions
    g.add_edge_list(edges)
    print(f'Data loaded in {time.time() - start_time:.2} seconds.')

    return g, gene_names


def check_genes_presence(complex_names, gene_names):
    '''Check whether all the protein subunits within a given complex
    are present in the CRISPR dataset
    Returns list removing non-present elements'''
    presence_dict = {gene: gene in gene_names for gene in complex_names}
    present_list = [key for key, value in presence_dict.items() if value]
    return present_list


def count_external_edges(g, internal_mask, external_mask):
    '''Counts edges external to a complex (i.e., connecting an external and an internal node)'''
    internal_view = gt.GraphView(g, vfilt=internal_mask)
    external_view = gt.GraphView(g, vfilt=external_mask)

    return g.num_edges() - internal_view.num_edges() - external_view.num_edges()


def edge_density_ratio(g, internal_mask, external_mask):
    '''Due to how this function computes the edge density ratio (EDR), the only way in which
    ZeroDivisionError can arise is if the node complex only consists of one single node. However,
    single-node complexes lack significance for our purpose and will be filtered out before calling
    this function'''
    # Count internal edges with a GraphView instantiation of the complex under consideration
    subgraph = gt.GraphView(g, vfilt=internal_mask)
    n_internal_edges = subgraph.num_edges()
    # Compute internal edge density (IED)
    possible_internal_edges = sum(internal_mask.get_array()) * (sum(internal_mask.get_array()) - 1) / 2
    ied = n_internal_edges / possible_internal_edges

    # Count external edges
    n_external_edges = count_external_edges(g, internal_mask, external_mask)
    # Compute external edge density (EED)
    possible_external_edges = sum(internal_mask.get_array()) * sum(external_mask.get_array())
    eed = n_external_edges / possible_external_edges

    # Return edge density ratio = IED / EED
    return ied / eed if eed != 0 else float('inf')


def single_rewiring(internal_nodes, external_nodes, degrees, progress_list):
    '''internal_nodes: list of indices of nodes internal to the complex;
    external_nodes: list of external nodes indices;
    degrees: dictionary of each node's index and its degree for the whole graph;
    progress_list: manager.list() object
    returns: the edge density ratio of one random rewiring'''
    # Get the node degree sequence
    current_degrees = degrees.copy()

    # Initialise internal & external edge counts
    N_int = 0
    N_ext = 0

    # 'internal_degrees_list' will contain the degrees of the internal nodes. Additionally,
    # its last element will be the sum of degrees of all external nodes
    internal_degrees_list = []
    for node in internal_nodes:
        internal_degrees_list.append(current_degrees[node])
    internal_degrees_list.append(sum(current_degrees[i] for i in external_nodes))

    # Connect stubs randomly, node by node
    for idx, node in enumerate(internal_nodes):
        # Do not change nodes until the current one is fully connected
        while internal_degrees_list[idx] > 0:
            # 'probs' will have the same length as 'internal_degrees_list' and will contain the probabilities
            # of each stub connecting to each of the internal nodes and any of the external nodes
            probs = [internal_degrees_list[i] / sum(internal_degrees_list[j] for j, _ in enumerate(internal_degrees_list) if j != idx) for i, _ in enumerate(internal_degrees_list) if i != idx]

            # Choose where to connect the stub based on 'probs' by drawing a uniformly distributed random number
            r = np.random.random()
            cum = probs[0]
            iteration = 1
            while r > cum:
                cum += probs[iteration]
                iteration += 1

            # If internal edge
            if iteration != len(probs):
                # Count the edge
                N_int += 1
                # Update 'internal_degrees_list'
                if iteration - 1 >= idx:
                    internal_degrees_list[iteration] -= 1
                elif iteration - 1 < idx:
                    internal_degrees_list[iteration - 1] -= 1
            # Else external edge
            else:
                # Count the edge
                N_ext += 1
                # Update 'internal_degrees_list'
                internal_degrees_list[-1] -=1
            internal_degrees_list[idx] -= 1





    # Calculate EDR for this iteration
    possible_internal_edges = len(internal_nodes) * (len(internal_nodes) - 1) / 2
    possible_external_edges = len(internal_nodes) * len(external_nodes)

    ied = N_int / possible_internal_edges
    eed = N_ext / possible_external_edges
    edr = ied / eed if eed != 0 else float('inf')

    progress_list.append(1)
    return edr


def simulate_rewiring(g, internal_nodes, num_iterations=1000):
    '''
    :param g: graph-tool Graph object
    :param internal_vertices: list of node indices that are internal to the complex
    :param num_iterations: number of rewirings
    :return: Observed edge density ratio (EDR) for the complex given g,
             p value of the observed EDR against the null distribution
             ratios: list containing the EDR of the null distribution samples
    '''
    # Get list of external nodes to call 'single_rewiring()'
    n_total = g.num_vertices()
    set_internal = set(internal_nodes)
    set_external = set(range(n_total)) - set_internal
    external_nodes = list(set_external)

    # Keep track of original degree sequence
    degrees = {int(v): v.out_degree() for v in g.vertices()}

    # Call 'single_rewiring()' parallelising the process
    with Manager() as manager:
        progress_list = manager.list()
        with Pool(processes=32, initializer=init_worker) as pool:
            results = [pool.apply_async(single_rewiring, args=(internal_nodes, external_nodes, degrees, progress_list)) for _ in range(num_iterations)]

            # Get results from all iterations
            ratios = [result.get() for result in results]

    # Create masks needed to call 'edge_density_ratio()'
    internal_mask = g.new_vertex_property('bool')
    external_mask = g.new_vertex_property('bool')
    for v in g.vertices():
        idx = int(v)
        internal_mask[v] = idx in internal_nodes
        external_mask[v] = idx in list(set_external)

    # Get observed EDR
    observed = edge_density_ratio(g, internal_mask, external_mask)

    # Calculate the p value
    p_value = np.mean([r >= observed for r in ratios])
    return observed, p_value, ratios


def edge_density(g):
    '''Returns global edge density of a Graph instance'''
    return g.num_edges() / (g.num_vertices() * (g.num_vertices() - 1) / 2)


def plot_density_per_threshold(n_points=40):
    # Array of thresholds to plot
    thresholds = np.arange(0, 1, 1 / n_points)

    # Parallelise the creation of the graph objects
    with Pool(processes=40) as pool:
        results = [pool.apply_async(prep_graph, args=(threshold)) for threshold in thresholds]
    gs = [result.get()[0] for result in results]

    # Obtain the density of each graph
    densities = [edge_density(g) for g in gs]

    # Plotting settings
    plt.rc('text', usetex=True)
    plt.rc('font', family='serif')
    plt.rcParams.update({'font.size': 21})

    # Plot
    ax, fig = plt.subplots(1, 1, figsize=(8, 6))
    ax.plot(thresholds, densities)
    ax.set_title('Edge density vs threshold')
    ax.set_xlabel('Threshold')
    ax.set_ylabel('Edge density')
    ax.grid(True)
    plt.tight_layout()
    plt.show()


def hist_num_genes(threshold):
    # Plotting settings
    plt.rc('text', usetex=True)
    plt.rc('font', family='serif')
    plt.rcParams.update({'font.size': 21})

    # Open the relevant results dataset
    number = str(threshold)[2:]
    df = pd.read_csv(f'results_{number}.csv')

    significant_df = df[df['Significant']]
    significant = significant_df['Number of genes'].values
    bins_significant = int(max(significant) - min(significant))

    nonsignificant_df = df[df['Significant'] == 0]
    nonsignificant = nonsignificant_df['Number of genes'].values
    bins_nonsignificant = int(max(nonsignificant) - min(nonsignificant))

    fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    ax.hist(nonsignificant, bins=bins_nonsignificant, color='blue', alpha=0.4, label='Non-significant complexes')
    ax.hist(significant, bins=bins_significant, color='orange', alpha=0.6, label='Significant complexes')
    ax.set_title(f'Histogram of number of genes (threshold = {threshold})')
    ax.set_xlabel('Number of genes')
    ax.set_ylabel('Frequency')
    ax.set_yscale('log')
    ax.xaxis.set_ticks(np.arange(min(min(significant), min(nonsignificant)), max(max(significant), max(nonsignificant))+1, 4))
    ax.legend()

    plt.tight_layout()
    plt.show()


def fraction_found_complexes(threshold):
    # Open the relevant results dataset
    number = str(threshold)[2:]
    df = pd.read_csv(f'results_{number}.csv')

    # Obtain fraction of successfully found complexes
    return len(df[df['Significant']]) / len(df)
