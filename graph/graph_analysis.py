import os
import numpy as np
import pandas as pd
from tqdm import tqdm
from data_processing.corum import load_CORUM
from statsmodels.stats.multitest import multipletests
from graph_utils import create_graph_gt, check_genes_presence, simulate_rewiring


def process_complex(complex, gene_names, g, num_iterations=10000):
    """
    Processes a CORUM complex: checks gene presence in CRISPR, simulates rewiring, and calculates EDR.

    Parameters
    ----------
    complex : list of str
    	List of gene names in the complex.
     gene_names : np.array
     	Array of gene names from the CRISPR dataset.
     g : graph_tool.Graph
     	The constructed gene dependency graph.
     num_iterations : int, optional
     	Number of random rewirings for null distribution. Default is 10000.

    Returns
    -------
    tuple
    Observed EDR, p-value, null distribution, number of genes in the complex, and presence flag.

    Notes
    -----
    Returns 5-tuples of NaNs if less than two subunits from the CORUM complex feature in the
    CRISPR dataset.
    """
    names = check_genes_presence(complex, gene_names)

    # Check if all genes are present in the CRISPR dataset
    all_genes_present = int(len(names) == len(complex))
    if len(names) <= 1:
        return (np.nan, np.nan, np.nan, np.nan, all_genes_present)

    # Get list of indices of internal nodes
    internal_nodes = [idx for idx, name in enumerate(gene_names) if name in names]

    # Simulate rewirings and compute the EDR for this complex, its p-value, and the null distribution of EDRs
    edr, p_value, null_distr = simulate_rewiring(g, internal_nodes, num_iterations)

    return (edr, p_value, null_distr, len(names), all_genes_present)


def process_all_complexes(threshold=0.20, num_iterations=10000):
    """
    Loads the CORUM complexes and CRISPR dataset, constructs the graph, and processes all complexes.

    This function requires that `get_abs_corrs()` and `filter_CORUM()` are run first. These functions
    are located in `../general/utils.py`.

    Parameters
    ----------
    threshold : float, optional
    	The correlation threshold for edge filtering in the graph. Default is 0.20.
    num_iterations : int, optional
    	Number of rewiring iterations for null distribution. Default is 10000.

    Returns
    -------
    df : pd.DataFrame
    	A dataframe containing processed results for each CORUM complex of interest. Columns:

     	- complexID (int): Unique identifier for the complex.
        - cell_line (str): The cell line associated with the complex.
	- subunits(Gene name) (str): Names of genes in the complex.
 	- pval (float): Raw p-value.
  	- corrected_pval_by (float): Adjusted p-value after Benjamini-Yekutieli (BY) false discover rate.
   	- significant_by (bool): True if the BY p-value is below the 0.05 significance threshold.
  	- corrected_pval_bh (float): Adjusted p-value after Benjamini-Hochberg (BH) false discover rate.
   	- significant_bh (bool): True if the BH p-value is below the 0.05 significance threshold.
    	- observed_edr (float): Computed EDR for the complex.
     	- num_genes (int): Number of genes present in the dataset for this complex.
      	- all_genes_present (bool): True if all genes in the CORUM complex are found in the CRISPR dataset.
       	- null_ratios (str): A string representation of a list with the null distribution of EDR values.

    Raises
    ------
    FileNotFoundError
        If the files `../datasets/abs_corrs.csv` and `../datasets/filtered_complexes.csv` are not found.
    """
    # Check if the preprocessed complexes and correlations exist
    missing_files = [
        f for f in [
            "../datasets/abs_corrs.csv",
            "../datasets/filtered_complexes.csv"
        ] if not os.path.exists(f)
    ]
    if missing_files:
        raise FileNotFoundError(
            f"Required preprocessing steps missing.\n"
            f"Run `get_abs_corrs()` and `filter_CORUM()` from `../general/utils.py` first.\n"
            f"Missing files: {missing_files}"
        )

    # Load dataset and preprocessed complexes
    df = pd.read_csv("../datasets/filtered_complexes.csv")
    complexes = load_CORUM()

    # Create graph from rank-normalised correlation matrix
    g = create_graph_gt(threshold, ranked=False)
    gene_names = np.array(g.vertex_properties["names"])

    # Prepare lists to store results
    results = [process_complex(complex, gene_names, g, num_iterations) for complex in tqdm(complexes)]

    # Unpack results
    observed_edrs, p_values, null_distributions, n_genes, all_genes = zip(*results)

    # Add results to dataframe
    df["all_genes_present"] = all_genes
    df["observed_edr"] = observed_edrs
    df["pval"] = p_values
    df["null_ratios"] = null_distributions
    df["num_genes"] = n_genes

    # Remove complexes that couldn't be processed
    df = df.dropna()

    # Apply multiple hypothesis corrections
    reject_by, p_corrected_by, _, _ = multipletests(df["pval"].values, alpha=0.05, method="fdr_by")
    df["significant_by"] = reject_by
    df["corrected_pval_by"] = p_corrected_by
    reject_bh, p_corrected_bh, _, _ = multipletests(df["pval"].values, alpha=0.05, method="fdr_bh")
    df["significant_bh"] = reject_bh
    df["corrected_pval_bh"] = p_corrected_bh

    # Reorder columns
    df = df[["complexID", "cell_line", "subunits(Gene name)", "pval", "corrected_pval_by", "significant_by",
             "corrected_pval_bh", "significant_bh", "observed_edr", "num_genes", "all_genes_present", "null_ratios"]]
    return df


def save_results(df, filename="results_abs/results_20.csv"):
    """
    Saves the dataframe to a CSV file.

    Parameters
    ----------
    df : pd.DataFrame
    	The dataframe containing processed results.
    filename : str, optional
    	The output filename. Default is "results_abs/results_20.csv", where the "20"
     	is referencing the correlation threshold value used to create the graph.
    """
    df.to_csv(filename, index=False)
    print(f"Results saved to {filename}")


if __name__ == "__main__":
    df_results = process_all_complexes(threshold=0.20, num_iterations=10000)
    save_results(df_results)
