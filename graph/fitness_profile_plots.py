import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist, squareform
from scipy.cluster.hierarchy import linkage, dendrogram
import matplotlib.pyplot as plt
import re

def preprocess_gene_data(
        filepath="../datasets/CRISPRGeneDependency.csv",
        threshold=0.3,
        random_sample=True,
        sample_size=10):
    """
    Loads and preprocesses gene dependency data for dendrogram analysis.

    Parameters
    ----------
    filepath : str, optional
        Path to the gene dependency dataset. Default is `../datasets/CRISPRGeneDependency.csv`
    threshold : float, optional
        Minimum essentiality threshold for gene selection. Default is 0.3.
    random_sample : bool, optional
        Whether to randomly sample genes. Default is True.
    sample_size : int, optional
        Number of genes to randomly sample. Default is 10.

    Returns
    -------
    df_filtered : pd.DataFrame
        Filtered dataframe with selected genes.
    list
        ist of gene names included
    """

    df = pd.read_csv(filepath, delimiter=",").iloc[:, 1:]  # Remove cell line names

    def apply_threshold(col): return col.max() > threshold

    df = df.iloc[:, df.apply(apply_threshold, axis=0).values]

    names = pd.read_csv("../datasets/names.txt", header=None).values.flatten()
    if random_sample:
        partial_names = list(np.random.choice(names, sample_size))
    else:
        # Genes belonging to a protein complex present in eukaryotic mitochondria acting on human heart tissue
        partial_names = ["ATP5PD", "ATP5MG", "MT-ATP6", "MT-ATP8",
                         "ATP5MC1", "ATP5F1B", "ATP5PF", "ATP5PB",
                         "ATP5F1A", "ATP5F1D", "ATP5F1C", "ATP5PO",
                         "ATP5MF", "ATP5F1E", "ATP5ME", "ATP5IF1"]

    def clean_names(col_name):
        """
        Takes a column name of type ABC123 (123) and strips the numbers in parenthesis out.
        """
        return re.sub(r"\s+\(\d+\)", "", col_name)

    regex_pattern = r"\b(" + "|".join(re.escape(label) for label in partial_names) + r")(?=\s|\Z)"
    df_filtered = df.filter(regex=regex_pattern).iloc[:, :12]
    df_filtered.columns = [clean_names(col) for col in df_filtered.columns]

    return df_filtered, df_filtered.columns.tolist()


def plot_dendrogram(df_filtered):
    """
    Computes hierarchical clustering and plots a dendrogram.

    Parameters
    ----------
    df_filtered : pd.DataFrame
        Processed gene dependency dataset.

    Returns
    -------
    dendro : dict
        Dendrogram object with clustering results
    """
    corrs = df_filtered.corr().fillna(0)    # Correlations
    np.fill_diagonal(corrs.values, 1)

    dists = 1 - corrs.abs()                 # High correlation implies short distance
    dists = np.clip(dists, 0, None)         # Correct for numerical errors making distances negative

    condensed_dists = squareform(dists)     # Convert to condensed form
    linked = linkage(condensed_dists,       # Perform the hierarchical clustering
                     method="complete")

    # Plot
    fig, ax = plt.subplots(figsize=(11, 12))
    dendro = dendrogram(linked, labels=df_filtered.columns, orientation="left", ax=ax)

    for label in ax.get_yticklabels():
        label.set_size(14)

    ax.set_xlabel("Distance", fontsize=36)
    plt.show()

    return dendro


def plot_fitness_profiles(df_filtered, dendro):
    """
    Plots fitness prfiles of genes in the order determined by the dendrogram.

    Parameters
    ----------
    df_filtered : pd.DataFrame
        Processed gene dependency dataset.
    dendro : dict
        Dendrogram object with gene order.
    """
    data = df_filtered.iloc[:, dendro["leaves"]].T  # Order as in dendrogram
    num_genes = data.shape[0]

    fig, ax = plt.subplots(num_genes, 1, figsize=(10, 12))

    for i, (idx, row) in enumerate(data.iterrows()):
        ax[num_genes - 1 - i].plot(row.values, label=idx)
        ax[num_genes - 1 - i].set_xticks([])
        ax[num_genes - 1 - i].set_yticks([])
        if i == 0:
            ax[num_genes - 1 - i].set_xlabel("Cell lines", fontsize=36)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    # Load and preprocess data
    df_filtered, selected_genes = preprocess_gene_data()

    # Plot dendrogram
    dendro = plot_dendrogram(df_filtered)

    # Plot fitness profiles
    plot_fitness_profiles(df_filtered, dendro)
