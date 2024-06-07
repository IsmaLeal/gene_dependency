import numpy as np
import pandas as pd
from tqdm import tqdm
from statsmodels.stats.multitest import multipletests
from graph_functions import get_genes, prep_graph, check_genes_presence, simulate_rewiring

## THIS FILE ASSUMES THAT graph_functions.get_ranked_corrs() and
# graph_functions.filter_CORUM() have already been run

# Load dataframe of complexes of interest
df = pd.read_csv('filtered_complexes.csv')

# 'complex_subunits': list of lists containing the gene names for all the genes in each protein complex
complex_subunits = df['subunits(Gene name)'].values
complexes = []
for idx, complex in enumerate(complex_subunits):
    complexes.append(get_genes(complex))    # 'get_genes()' returns a list of gene names given a complex

# Create graph from rank-normalised correlation matrix
threshold = 0.98
g, gene_names = prep_graph(threshold)

# Prepare lists to hold results (will be added as columns to the original filtered dataframe)
observed_edrs = []
p_values = []
null_distributions = []
all_genes = np.ones_like(complex_subunits)  # Boolean column (0 if any gene from the CORUM complex is not in the CRISPR dataset)
n_genes = []

# Iterate through each complex
for idx, complex in tqdm(enumerate(complexes)):
    # Check if all genes are present in the CRISPR dataset
    names = check_genes_presence(complex, gene_names)
    if len(names) != len(complex):
        # The filtered dataframe has had its single-gene complexes removed, but need to recheck
        # if at least one of its components has not been found in the CRISPR dataset
        all_genes[idx] = 0  # Not all genes from the complex are present in CRISPR
        if len(names) <= 1:
            # If only 1 or none of the genes from the complex have been found, discard this complex
            observed_edrs.append(np.nan)
            p_values.append(np.nan)
            null_distributions.append(np.nan)
            n_genes.append(np.nan)
            continue
    # Get list of indices of internal nodes to call 'simulate_rewiring()'
    internal_nodes = [idx for idx, name in enumerate(gene_names) if name in names]
    # Obtain edge density ratio, null distribution and p-value for the complex
    edr, p_value, null_distr = simulate_rewiring(g, internal_nodes, 1000)
    # Save obtained values
    observed_edrs.append(edr)
    p_values.append(p_value)
    null_distributions.append(null_distr)
    n_genes.append(len(names))

# Apply Benjamini-Yekutieli False Discovery rate with cutoff of 0.05 to obtain the significant p-values
# 'reject' is a Boolean mask True if a p-value is considered significant
# 'alpha_corrected' returns 8.56e-5 for a threshold of 0.9
reject, _, _, alpha_corrected = multipletests(np.array(p_values), alpha=0.05, method='fdr_by')

# Add results to dataframe
df['All genes from CORUM present in CRISPR'] = all_genes
df['Observed density ratio'] = observed_edrs
df['p-value'] = p_values
df['Null distribution (1000 samples)'] = null_distributions
df['Number of genes'] = n_genes
df['Significant'] = reject

# Those CORUM complexes with 0 or 1 genes in the CRISPR data frame
# Return NaN for the EDR, p-values, and null distribution. Remove them from the
# data frame
df = df.dropna()

# Save as .csv file
df.to_csv('results_98.csv', index=False)