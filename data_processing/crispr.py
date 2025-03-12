import numpy as np
import pandas as pd
from tqdm import tqdm

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
        temp = arr.argsort()  # Indices that sort the array
        ranks = temp.argsort() + 1  # Ranks

        # Compute rank of ties
        sorted = np.sort(arr)
        sorted_ranks = np.sort(ranks).astype('float64')
        for i in range(len(arr)):
            start = 0
            if i == 0 or sorted[i] != sorted[i - 1]:  # Check if element starts a series of ties
                start = i
            if i == len(arr) - 1 or sorted[i] != sorted[i + 1]:  # If it ends a series of ties
                end = i + 1
                avg_rank = np.mean(sorted_ranks[start:end])
                sorted_ranks[start:end] = avg_rank

        return sorted_ranks[np.argsort(temp)]

    df = pd.read_csv("../datasets/CRISPRGeneDependency.csv", delimiter=",")
    depmap = df.iloc[:, 1:]  # Get rid of cell line names
    gene_names = depmap.columns.values  # Save gene names as np.ndarray
    corrs_matrix = depmap.corr()  # Get correlation matrix
    corrs_matrix.fillna(0, inplace=True)  # Adjust for NaN due to some genes having S.D.=0
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
    matrix.to_csv("../datasets/ranked_corrs.csv", index=True)


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
    depmap = df.iloc[:, 1:]               # Get rid of cell line names
    gene_names = depmap.columns.values    # Save gene names as np.ndarray
    corrs_matrix = depmap.corr()          # Get correlation matrix
    corrs_matrix.fillna(0, inplace=True)  # Adjust for NaN due to some genes having S.D.=0
    np.fill_diagonal(corrs_matrix.values, 0)
    corrs_matrix.columns = gene_names
    corrs_matrix.index = gene_names
    abs_corrs = corrs_matrix.abs()        # Absolute values
    abs_corrs.to_csv("../datasets/abs_corrs.csv", index=True)


if __name__ == "__main__":
    get_abs_corrs()
