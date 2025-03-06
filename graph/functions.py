import os
import graph_tool.all as gt
import pandas as pd
import numpy as np
import time
from typing import Tuple, List
from tqdm import tqdm
import matplotlib.pyplot as plt
from statsmodels.stats.multitest import multipletests
from multiprocessing import Pool, Manager


def get_ranked_corrs() -> None:
    """
    Computes and saves a symmetric matrix of ranked correlations for every pair of genes.

    The function loads a gene dependency dataset, computes the correlation matrix, applies
    a ranking transformation to each row, and symmetrises the result.
    The array is saved in 'rank_transf_symm.csv' with row and column labels.

    Notes
    -----
    - The ranking transformation accounts for ties by assigning their average rank.
    - The correlation matrix is adjusted for NaN values by replacing them with zero.
    - The symmetrisation ensures the highest rank is preserved between pairs.
    """
    def rank_array(arr: np.ndarray) -> np.ndarray:
        """
        Ranks the elements in an array, handling ties by assigning their average rank.

        Parameters
        ----------
        arr : np.ndarray
            Input numerical array.

        Returns
        -------
        np.ndarray
            Ranked array with the same shape as the input.
        """
        temp = arr.argsort()            # Indices that sort the array
        ranks = temp.argsort() + 1      # Ranks

        # Compute rank of ties
        sorted = np.sort(arr)
        sorted_ranks = np.sort(ranks).astype('float64')
        for i in range(len(arr)):
            start = 0
            if i == 0 or sorted[i] != sorted[i - 1]:            # Check if element starts a series of ties
                start = i
            if i == len(arr) - 1 or sorted[i] != sorted[i + 1]: # If it ends a series of ties
                end = i + 1
                avg_rank = np.mean(sorted_ranks[start:end])
                sorted_ranks[start:end] = avg_rank
                
        return sorted_ranks[np.argsort(temp)]

    df = pd.read_csv("../datasets/CRISPRGeneDependency.csv", delimiter=",")
    depmap = df.iloc[:, 1:]                       # Get rid of cell line names
    gene_names = depmap.columns.values            # Save gene names as np.ndarray
    corrs_matrix = depmap.corr()                  # Get correlation matrix
    corrs_matrix.fillna(0, inplace=True)          # Adjust for NaN due to some genes having S.D.=0
    np.fill_diagonal(corrs_matrix.values, 0)

    # Create array to hold the ranks
    rank_transformation = np.zeros_like(corrs_matrix.values)
    for idx, cell_line in tqdm(enumerate(corrs_matrix.values)):
        rank_transformation[idx, :] = rank_array(cell_line)

    # Symmetrise by taking the largest rank
    for i in tqdm(range(rank_transformation.shape[0])):
        for j in range(i, rank_transformation.shape[1]):
            if rank_transformation[i, j] > rank_transformation[j, i]:
                rank_transformation[j, i] = rank_transformation[i, j]
            elif rank_transformation[i, j] < rank_transformation[j, i]:
                rank_transformation[i, j] = rank_transformation[j, i]

    # Save as .csv file
    matrix = pd.DataFrame(rank_transformation)
    matrix.columns = gene_names
    matrix.index = gene_names
    matrix.to_csv("ranked_corrs_2.csv", index=True)


def get_abs_corrs() -> None:
    """
    Computes and saves a symmetric matrix of absolute correlations for every pair of genes.
    
    The function loads a gene dependency dataset and computes the absolute correlation matrix.
    The array is saved in 'rank_transf_symm.csv' with row and column labels.

    Notes
    -----
    - The correlation matrix is adjusted for NaN values by replacing them with zero.
    """
    df = pd.read_csv("../datasets/CRISPRGeneDependency.csv", delimiter=",")
    depmap = df.iloc[:, 1:]                   # Get rid of cell line names
    gene_names = depmap.columns.values        # Save gene names as np.ndarray
    corrs_matrix = depmap.corr()              # Get correlation matrix
    corrs_matrix.fillna(0, inplace=True)      # Adjust for NaN due to some genes having S.D.=0
    np.fill_diagonal(corrs_matrix.values, 0)
    corrs_matrix.columns = gene_names
    corrs_matrix.index = gene_names
    abs_corrs = corrs_matrix.abs()            # Absolute values
    abs_corrs.to_csv("../datasets/abs_corrs_2.csv", index=True)


def clean_col_names(col: str) -> str:
    """
    Extracts gene names from a label formatted as 'GeneName (GeneID)'.

    Parameters
    ----------
    col : str
        Column label containing gene name and ID.

    Returns
    -------
    str
        Extracted gene name.
    """
    return col.split(" (")[0]


def get_genes(complex: str) -> List[str]:
    """
    Splits a complex string containing multiple gene names separated by ';'.

    Example input: 'PRORP;TRMT10C;HSD17B10'
    Example output: ['PRORP', 'TRMT10C', 'HSD17B10']
    
    Parameters
    ----------
    complex : str
        String of gene names separated by semicolons.

    Returns
    -------
    List[str]
        List of individual gene names.
    """
    return complex.split(";")


def filter_CORUM() -> pd.DataFrame:
    """
    Filters the CORUM dataset for relevant protein complexes based on pre-defined cell line names.

    Returns
    -------
    pd.DataFrame
        Filtered dataset of protein complexes.
    """
    # Load CORUM dataset
    file_path = "../datasets/humanComplexes.txt"
    try:
        df = pd.read_csv(file_path, delimiter="\t")
    except:
        print(f"File {file_path} does not exist. Please re-download from CORUM website")

    # Select rows containing these substrings in their 'Cell line' value
    substrings = ["T cell line ED40515",
                  "mucosal lymphocytes",
                  "CLL cells",
                  "monocytes",
                  "THP-1 cells",
                  "bone marrow-derived",
                  "monocytes, LPS-induced",
                  "THP1 cells",
                  "human blood serum",
                  "human blood plasma",
                  "plasma",
                  "CSF",
                  "human leukemic T cell JA3 cells",
                  "erythrocytes",
                  "peripheral blood mononuclear cells",
                  "African Americans",
                  "SKW 6.4",
                  "BJAB cells",
                  "Raji cells",
                  "HUT78",
                  "J16",
                  "H9",
                  "U-937",
                  "Jurkat T",
                  "NB4 cells",
                  "U937",
                  "early B-lineage",
                  "T-cell leukemia",
                  "lymphoblasts",
                  "whole blood and lymph",
                  "human neutrophil-differentiating HL-60 cells",
                  "human peripheral blood neutrophils",
                  "human neutrophils from fresh heparinized human peripheral blood",
                  "human peripheral blood",
                  "HCM",
                  "liver-hematopoietic",
                  "cerebral cortex",
                  "human brain",
                  "pancreatic islet",
                  "human hepatocyte carcinoma HepG2 cells",
                  "Neurophils",
                  "H295R adrenocortical",
                  "frontal cortex",
                  "myometrium",
                  "vascular smooth muscle cells",
                  "Dendritic cells",
                  "intestinal epithelial",
                  "Primary dermal fibroblasts",
                  "HK2 proximal",
                  "brain pericytes",
                  "HepG2",
                  "HEK 293 cells, liver",
                  "normal human pancreatic duct epithelial",
                  "pancreatic ductal adenocarcinoma",
                  "OKH cells",
                  "cultured podocytes",
                  "renal glomeruli",
                  "VSMCs",
                  "differentiated HL-60 cells",
                  "SH-SY5Y cells",
                  "frontal and entorhinal cortex",
                  "SHSY-5Y cells",
                  "hippocampal HT22 cells",
                  "primary neurons",
                  "neurons",
                  "renal cortex membranes",
                  "Kidney epithelial cells",
                  "skeletal muscle cells",
                  "Skeletal muscle fibers",
                  "differentiated 3T3-L1",
                  "brain cortex",
                  "cortical and hippocampal areas",
                  "human H4 neuroglioma",
                  "Thalamus",
                  "HISM",
                  "pancreas",
                  "RCC4",
                  "C2C12 myotube",
                  "XXVI muscle",
                  "SH-SY5Y neuroblastoma",
                  "HCC1143",
                  "Hep-2",
                  "PANC-1",
                  "HEK293T cells",
                  "HEK-293 cells",
                  "heart",
                  "epithelium",
                  "kidney",
                  "heart muscle",
                  "central nervous system",
                  "COS-7 cells",
                  "ciliary ganglion",
                  "striated muscle",
                  "PC12",
                  "293FR cells"]
    pattern = "|".join(substrings)

    # Select rows whose 'Cell line' value is exactly one of these
    exact = ["muscle", "293 cells", "brain", "HEK 293 cells"]

    # Create Boolean mask selecting all the rows described
    partial_mask = df["Cell line"].str.contains(pattern, case=False, na=False)
    exact_mask = df["Cell line"].isin(exact)
    total_mask = partial_mask | exact_mask

    # Obtain filtered dataframe, select relevant columns, sort by 'Cell line'
    complexes_full = df[total_mask]
    complexes = complexes_full[["ComplexID", "ComplexName", "Cell line", "subunits(Gene name)", "GO description", "FunCat description"]]
    complexes = complexes.sort_values(by=["Cell line"])

    # Exclude those complexes including only one subunit/ gene
    mask_mono = [len(complexes["subunits(Gene name)"].values[i].split(";")) > 1 for i in range(len(complexes))]
    complexes = complexes.loc[mask_mono]
    return complexes


def load_CORUM() -> List[str]:
    """
    Loads the filtered CORUM dataset from '../datasets/filtered_complexes.csv'.

    Returns
    -------
    complexes : List[str]
        List of protein complexes (each complex is a list of gene names) or None
        if the file is missing.
    
    Examples
    --------
    >>> complexes = load_CORUM()
    >>> print(complexes[0])
    ["GeneA", "GeneB", "GeneC"]
    """
    file_path = "../datasets/filtered_complexes.csv"
    try:
        df = pd.read_csv(file_path)
        complexes_strings = df['subunits(Gene name)'].values
        complexes = [complex.split(";") for complex in complexes_strings]
    except:
        print(f"File {file_path} does not exist. Please run filter_CORUM().")
        complexes = None
    return complexes


def init_worker() -> None:
    """
    Initialises a worker process by setting a unique random seed based on the process ID.

    Ensures that different processes generate different random sequences.
    """
    seed = os.getpid()
    np.random.seed(seed)


def prep_graph(threshold: float = 0.2, ranked: bool = False) -> graph_tool.Graph:
    """
    Constructs a graph-tool Graph object from a correlation matrix.

    The function loads either the ranked or absolute correlation matrix, normalises it,
    applies a threshold to create an adjacency matrix, and constructs an undirected simple
    graph where nodes represent genes.

    Parameters
    ----------
    threshold : float, optional
        Minimum correlation value to consider an edge. Has to be between 0 and 1. Defaults to 0.2.
    ranked : bool, optional
        Whether to use the ranked correlation matrix. Default is False.

    Returns
    -------
    graph_tool.Graph
        A graph object representing gene interactions.

    Examples
    --------
    >>> g = prep_graph(threshold=0.5, ranked=True)
    >>> print(g.num_vertices)
    """
    try:
        if ranked:
            corrs = pd.read_csv("../datasets/ranked_corrs.csv", delimiter=",", index_col=0)
        else:
            corrs = pd.read_csv("../datasets/abs_corrs.csv", delimiter=",", index_col=0)
        corrs /= np.max(corrs.values)   # Normalise
        gene_names = np.array(          # Remove gene IDs
            [clean_col_names(col) for col in corrs.columns]
        )
    
        # Create adjacency matrix A
        A = (corrs.values > threshold).astype(np.int8)
    
        # Instantiate graph-tool Graph and add nodes & edges based on A
        g = gt.Graph(directed=False)
        g.add_vertex(n=A.shape[0])
        edges = np.transpose(np.nonzero(np.triu(A, 1))) # Use k=1 to prevent self-interactions
        g.add_edge_list(edges)
    
        # Add gene names to the nodes
        names = g.new_vertex_property("string")
        for v in g.vertices():
            names[int(v)] = gene_names[int(v)]
        g.vertex_properties["names"] = names
        
    except FileNotFoundError as e:
        print(f"Error: {e}. Try running get_ranked_corrs() or get_abs_corrs().")

    return g


def check_genes_presence(complex_names: List[str],
                         gene_names: List[str]) -> List[str]:
    """
    Checks whether all the protein subunits within a given CORUM complex are present in the CRISPR dataset.

    Parameters
    ----------
    complex_names : List[str]
        List of gene names forming a given CORUM complex.
    gene_names : List[str]
        List of CRISPR gene names available.

    Returns
    -------
    present_list : List[str]
        List of genes from the CORUM complex that are present in the CRISPR dataset.
    """
    presence_dict = {gene: gene in gene_names for gene in complex_names}
    present_list = [key for key, value in presence_dict.items() if value]
    return present_list


def count_external_edges(g: graph_tool.Graph,
                         internal_mask: graph_tool.PropertyMap,
                         external_mask: graph_tool.PropertyMap) -> int:
    """
    Counts the number of edges that connect an internal node (part of a complex) to an external node.

    Parameters
    ----------
    g : graph_tool.Graph
        The graph-tool Graph object.
    internal_mask : graph_tool.PropertyMap
        Boolean mask for internal nodes.
    external_mask : graph_tool.PropertyMap
        Boolean mask for external nodes.

    Returns
    -------
    int
        Number of edges connecting a node in the complex to an external node.
    """
    internal_view = gt.GraphView(g, vfilt=internal_mask)
    external_view = gt.GraphView(g, vfilt=external_mask)

    return g.num_edges() - internal_view.num_edges() - external_view.num_edges()


def edge_density_ratio(g: graph_tool.Graph,
                       internal_mask: graph_tool.PropertyMap,
                       external_mask: graph_tool.PropertyMap) -> float:
    """
    Computes the Edge Density Ratio (EDR), which is the ratio of internal edge density (IED) to external edge density (EED).

    Parameters
    ----------
    g : graph_tool.Graph
        The graph-tool Graph object.
    internal_mask : graph_tool.PropertyMap
        Boolean mask for internal nodes.
    external_mask : graph_tool.PropertyMap
        Boolean mask for external nodes.

    Returns
    -------
    float
        The edge density ratio. Returns infinity if EED is zero.

    Notes
    -----
    ZeroDivisionError can only occur if the complex consists of a single node. These complexes
    are already filtered out by filter_CORUM().
    """
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
    return ied / eed if eed != 0 else float("inf")


def single_rewiring(internal_nodes: List[int],
                    external_nodes: List[int],
                    degrees: dict,
                    progress_list: multiprocessing.Manager().list) -> float:
    """
    Generates a randomised graph instance from the configuration model and calculates the Edge Density Ratio (EDR).

    The function simulates rewiring by randomly connecting internal nodes with each other or with external nodes
    based on their degree distribution. It ensures that each stub from an internal node connects to another node
    based on computed probabilities. The algorithm is:
    1. Initialise counters for internal (N_int) and external (N_ext) edges.
    2. Construct a list (`internal_degrees_list`) storing degrees of internal nodes and the sum
       of all external nodes' degrees.
    3. For each internal node:
        - While it has unconnected stubs, compute probabilities of connecting to other nodes.
        - Select a target node using a probability-weighted random draw.
        - If the target is internal, increase N_int; otherwise, increase N_ext.
        - Update degrees accordingly.
    4. Compute IED and EED.
    5. Return the EDR = IED / EED.

    Parameters
    ----------
    internal_nodes : List[int]
        Indices of nodes that belong to the complex of interest.
    external_nodes : List[int]
        Indices of nodes that are external to the complex.
    degrees : dict
        Dictionary mapping node indices to their respective degrees in the original graph.
    progress_list : multiprocessing.Manager().list
        A shared list used to track progress in parallel execution.

    Returns
    -------
    edr : float
        The computed EDR for this rewiring iteration.    
    """
    # Get the node degree sequence
    current_degrees = degrees.copy()

    # Initialise internal & external edge counts
    N_int = 0
    N_ext = 0

    # Create a list storing degrees of internal nodes, with the last element being the sum of external degrees
    internal_degrees_list = []
    for node in internal_nodes:
        internal_degrees_list.append(current_degrees[node])
    internal_degrees_list.append(sum(current_degrees[i] for i in external_nodes))

    # Iterate through internal nodes, assigning stubs probabilistically
    for idx, node in enumerate(internal_nodes):
        while internal_degrees_list[idx] > 0:
            # Compute probabilities for connecting stubs to other nodes
            probs = [internal_degrees_list[i] / sum(internal_degrees_list[j] for j, _ in enumerate(internal_degrees_list) if j != idx) for i, _ in enumerate(internal_degrees_list) if i != idx]

            # Select target node based on computed probabilities
            r = np.random.random()
            cum = probs[0]
            iteration = 1
            while r > cum:
                cum += probs[iteration]
                iteration += 1

            # Determine whether internal or external
            if iteration != len(probs):
                N_int += 1
                if iteration - 1 >= idx:
                    internal_degrees_list[iteration] -= 1
                elif iteration - 1 < idx:
                    internal_degrees_list[iteration - 1] -= 1
            else:
                N_ext += 1
                internal_degrees_list[-1] -=1
            internal_degrees_list[idx] -= 1

    # Calculate EDR for this iteration
    possible_internal_edges = len(internal_nodes) * (len(internal_nodes) - 1) / 2
    possible_external_edges = len(internal_nodes) * len(external_nodes)

    ied = N_int / possible_internal_edges
    eed = N_ext / possible_external_edges
    edr = ied / eed if eed != 0 else float("inf")

    progress_list.append(1)    # Track progress for multiprocessing
    return edr


def simulate_rewiring(g: graph_tool.Graph,
                      internal_nodes: List[int],
                      num_iterations: int = 1000) -> Tuple[float, float, List[float]]:
    """
    Calls `single_rewiring()` multiple times, constructing a null distribution for the Edge
    Density Ratio (EDR) and assessing the statistical significance of the observed EDR.

    Parameters
    ----------
    g : graph_tool.Graph
        The graph-tool Graph object representing the genes.
    internal_nodes : List[int]
        List of node indices that are internal to the complex of interest.
    num_iterations : int, optional
        Number of random rewirings to perform. Default is 1000.

    Returns
    -------
    observed : float
        Observed EDR for the given complex in the original graph.
    p_value : float
        Statistical significance of the observed EDR against the null distribution.
    ratios : List[float]
        Null distribution of EDR values from rewiring simulations.

    Examples
    --------
    >>> observed, p_value, ratios = simulate_rewiring(g, [0, 1, 2, 4], num_iterations=2000)
    """
    # Get list of external nodes to call `single_rewiring()`
    n_total = g.num_vertices()
    set_internal = set(internal_nodes)
    set_external = set(range(n_total)) - set_internal
    external_nodes = list(set_external)

    # Save original node degree sequence
    degrees = {int(v): v.out_degree() for v in g.vertices()}

    # Call `single_rewiring()` parallelising the process
    with Manager() as manager:
        progress_list = manager.list()
        with Pool(processes=40, initializer=init_worker) as pool:
            results = [pool.apply_async(single_rewiring, args=(internal_nodes, external_nodes, degrees, progress_list)) for _ in range(num_iterations)]
            ratios = [result.get() for result in results]

    # Create masks needed to call `edge_density_ratio()`
    internal_mask = g.new_vertex_property("bool")
    external_mask = g.new_vertex_property("bool")
    for v in g.vertices():
        idx = int(v)
        internal_mask[v] = idx in internal_nodes
        external_mask[v] = idx in list(set_external)

    # Compute observed EDR from original graph
    observed = edge_density_ratio(g, internal_mask, external_mask)
    ratios.append(observed)

    # Compute the p-value
    p_value = np.mean([r >= observed for r in ratios])
    
    return observed, p_value, ratios


def edge_density(g: graph_tool.Graph) -> float:
    """
    Computes the global edge density of a graph.

    Parameters
    ----------
    g : graph_tool.Graph
        Input graph-tool Graph object.

    Returns
    -------
    float
        Edge density, defined as the ratio of the number of edges over the total
        number of possible edges.
    """
    return g.num_edges() / (g.num_vertices() * (g.num_vertices() - 1) / 2)


def plot_edgestats_per_threshold(n_points: int = 100) -> None:
    """
    Plots edge density and number of edges as a function of the threshold used to construct the graph.

    The function iterates over different threshold values, constructs a graph for each, computes the edge
    density and number of edges, and then generates two plots: one for edge density and another for edge count.

    Parameters
    ----------
    n_points : int, optional
        Number of threshold values to test. Default is 100.
    """
    thresholds = np.arange(0, 1, 1 / n_points)

    # Parallelise the graph creation
    with Pool(processes=40) as pool:
        results = [pool.apply_async(prep_graph, args=(threshold, False)) for threshold in thresholds]
        gs = [result.get() for result in results]

    # Save statistics
    densities = [edge_density(g) for g in gs]
    numbers_edges = [g.num_edges() for g in gs]

    # Plotting settings
    plt.rc("text", usetex=True)
    plt.rc("font", family="serif")
    plt.rcParams.update({"font.size": 23})

    # Plot
    fig, ax = plt.subplots(1, 2, figsize=(14, 6))
    ax[0].plot(thresholds, densities)
    ax[0].set_title("Edge density vs threshold")
    ax[0].set_xlabel("Threshold")
    ax[0].set_ylabel("Edge density")
    ax[0].grid(True)

    ax[1].plot(thresholds, numbers_edges)
    ax[1].set_title("Edge number vs threshold")
    ax[1].set_xlabel("Threshold")
    ax[1].set_ylabel("Edge number")
    ax[1].set_yscale("log")
    ax[1].grid(True)

    plt.tight_layout()
    plt.show()


def hist_num_genes(threshold: float) -> None:
    """
    Plots a histogram of the number of genes per significant complex at a given threshold.

    This function loads a results file, extracts the number of genes for statistically significant
    and non-significant complexes, and generates a histogram.

    Parameters
    ----------
    threshold : float
        The correlation threshold used in the analysis.
    """
    # Plotting settings
    plt.rc("text", usetex=True)
    plt.rc("font", family="serif")
    plt.rcParams.update({"font.size": 21})

    # Open the relevant results dataset
    number = str(threshold)[2:]
    try:
        df = pd.read_csv(f"results/results_{number}.csv")
    except FileNotFoundError as e:
        print(f"Error: The file 'results/results_{number}.csv' was not found.")
        print("Please run 'fraction_complexes.py' in this directory to generate the required file.")

    # Extract gene counts
    significant_df = df[df["Significant (BY)"]]
    significant = significant_df["# genes"].values
    bins_significant = int(max(significant) - min(significant))

    nonsignificant_df = df[df["Significant (BY)"] == 0]
    nonsignificant = nonsignificant_df["# genes"].values
    bins_nonsignificant = int(max(nonsignificant) - min(nonsignificant))

    # Create histogram
    fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    ax.hist(nonsignificant, bins=bins_nonsignificant, color="blue", alpha=0.4, label="Non-significant complexes")
    ax.hist(significant, bins=bins_significant, color="orange", alpha=0.6, label="Significant complexes")
    ax.set_title(f"Histogram of number of genes (threshold = {threshold})")
    ax.set_xlabel("Number of genes")
    ax.set_ylabel("Frequency")
    ax.set_yscale("log")
    ax.xaxis.set_ticks(np.arange(min(min(significant), min(nonsignificant)), max(max(significant), max(nonsignificant))+1, 4))
    ax.legend()

    plt.tight_layout()
    plt.show()


def fraction_found_complexes(threshold: float) -> float:
    """
    Computes the fraction of complexes identified as significant for a given threshold.

    The function loads a dataset corresponding to the given threshold and calculates the
    fraction of complexes that have been marked as statistically significant.

    Parameters
    ----------
    threshold : float
        The correlation threshold used for filtering significant edges in the creation of the graph.

    Returns
    -------
    float
        The fraction of complexes that were found to be significant.
    """
    number = str(threshold)[2:]
    df = pd.read_csv(f"results_abs/results_{number}.csv")

    return len(df[df["Significant"]]) / len(df)
