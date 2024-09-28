import numpy as np
import pandas as pd
from tqdm import tqdm
from statsmodels.stats.multitest import multipletests
from functions import prep_graph, load_CORUM, check_genes_presence, simulate_rewiring

'''
THIS FILE ASSUMES THAT graph/functions.get_ranked_corrs() and
graph/functions.filter_CORUM() have already been run

WARNING: prep_graph() returns a graph_tool.Graph class object
	NetworkX is only used in hypergraph/ and ml_model/
'''

# Load dataframe of pre-processed complexes of interest
df = pd.read_csv('../datasets/filtered_complexes.csv')

# 'complexes': list of lists containing the gene names of subunits of each complex
complexes = load_CORUM()

# Create graph from rank-normalised correlation matrix
threshold = 0.20
g = prep_graph(threshold, ranked=False)
gene_names = np.array(g.vertex_properties['names'])

# Prepare lists to hold results (will be added as columns to the filtered dataframe)
observed_edrs = []
p_values = []
null_distributions = []
all_genes = np.ones_like(complexes)  # Boolean column (0 if any gene from the CORUM complex is not in the CRISPR dataset)
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
    edr, p_value, null_distr = simulate_rewiring(g, internal_nodes, 10000)
    # Save obtained values
    observed_edrs.append(edr)
    p_values.append(p_value)
    null_distributions.append(null_distr)
    n_genes.append(len(names))

# Apply Benjamini-Yekutieli False Discovery rate with cutoff of 0.05 to obtain the significant p-values
# 'reject' is a Boolean mask True if a p-value is considered significant
# 'alpha_corrected' returns 8.56e-5 for a threshold of 0.9

# Add results to dataframe
df['All genes present'] = all_genes
df['Observed ratio'] = observed_edrs
df['pval'] = p_values
df['Null ratios'] = null_distributions
df['# genes'] = n_genes

# Those CORUM complexes with 0 or 1 genes in the CRISPR data frame
# Return NaN for the EDR, p-values, and null distribution. Remove them from the
# data frame
df = df.dropna()

reject, p_corrected, _, alpha_corrected = multipletests(df['pval'].values, alpha=0.05, method='fdr_by')
df['Significant'] = reject
df['Corrected pval'] = p_corrected

df = df[['ComplexID', 'Cell line', 'subunits(Gene name)', 'pval', 'Corrected pval',
         'Significant', 'Observed ratio', '# genes', 'All genes present', 'Null ratios']]

df = adapt_df(df)

# Save as .csv file
df.to_csv('results_abs/results_15.csv', index=False)
