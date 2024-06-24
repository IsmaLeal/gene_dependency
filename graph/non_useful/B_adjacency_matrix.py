import pandas as pd
import numpy as np
import time

def create_adjacency_matrix(threshold):
    # Load ranked-normalised correlations matrix (gene names are column labels)
    start = time.time()
    corrs = pd.read_csv('../rank_transf_symm_2.csv', delimiter=',', index_col=0)
    end = time.time()
    times = end - start
    print(f'Time taken to open .csv file: {int(times/60)}min {int(times%60)}s')
    corrs /= np.max(corrs.values)

    # Get gene names
    gene_names = corrs.columns.values

    # Set threshold and obtain adjacency matrix A
    threshold = threshold
    A_np = np.where(corrs.values > threshold, 1, 0)
    A = pd.DataFrame(A_np)
    A.index = gene_names
    A.columns = gene_names

    return A
