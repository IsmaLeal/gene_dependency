import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist, squareform
from scipy.cluster.hierarchy import linkage, dendrogram
import matplotlib.pyplot as plt
import re

'''
THIS FILE PRODUCES 2 PLOTS OF 36 SELECTED GENES
(i) dendrogram of distances, where the distance is 1 - pearson_correlation. The correlations were obtained
for every pair of genes over all the cell lines for which both were screened (ie, excluding NaNs)
(ii) fitness profile for each gene in the dendrogram

Together (dendrogram left, plots right) they look better
'''

# Plotting settings
plt.rc('text', usetex=True)         # Allow LaTeX
plt.rc('font', family='arial')      # Font
plt.rcParams.update({'font.size': 15})    # Fontsize

# Data processing
df = pd.read_csv('../datasets/CRISPRGeneDependency.csv', delimiter=',')    # Load dataframe
df = df.iloc[:, 1:]     # Remove cancer cell line names
df = df ** 3            # Raise to the 3rd power to highlight essentiality
threshold = 0.3         # Less essential genes will be omitted

def apply_threshold(col): return col.max() > threshold  # Returns Boolean describing if exceeds threshold or not

df = df.iloc[:, df.apply(apply_threshold, axis=0).values]  # Only select columns exceeding threshold
a = df.columns # Column names


# Filter for the 36 genes of interest
#partial_names = ['ATP5PD', 'ATP5MG', 'MT-ATP6', 'MT-ATP8', 'ATP5MC1', 'ATP5F1B'
#                 'ATP5PF', 'ATP5PB', 'ATP5F1A', 'ATP5F1D', 'ATP5F1C', 'ATP5PO',
#                 'ATP5MF', 'ATP5F1E', 'ATP5ME', 'ATP5IF1']
names = pd.read_csv('../datasets/names.txt', header=None)
partial_names = list(np.random.choice(names.values.flatten(), 100))


def clean_names(col_name):
    '''
    Takes a column name of type ABC123 (123) and strips the numbers in parentheses out of it
    '''
    name = re.sub(r'\s+\(\d+\)', '', col_name)
    return name


# Regex pattern for columns of interest
regex_pattern = r'\b(' + '|'.join(re.escape(label) for label in partial_names) + r')(?=\s|\Z)'
# Filter columns of interest
df_filtered = df.filter(regex=regex_pattern)
df_filtered = df_filtered.iloc[:, :12]
# Remove parentheses from the gene names
df_filtered.columns = [clean_names(col) for col in df_filtered.columns]


# Apply correlation to all pairs of genes (columns)
corrs = df_filtered.corr()
corrs.fillna(0, inplace=True)           # Avoid NaN values if some gene has SD=0
np.fill_diagonal(corrs.values, 1)   # Genes with SD=0 will have a correlation with themselves of 0, correct to 1

# Obtain the distance as 1 - |correlation|
dists = 1 - corrs.abs()
dists = np.clip(dists, 0, None) # Correct for numerical errors making distances negative
# Convert distance matrix to a condensed form
condensed_dists = squareform(dists)
# Hierarchical clustering
linked = linkage(condensed_dists, method='complete')

# Plot dendrogram
fig_dendro, ax_dendro = plt.subplots(1, 1, figsize=(11, 12))
dendro = dendrogram(linked, labels=df_filtered.columns, orientation='left', distance_sort='descending', ax=ax_dendro)

# Plot ordered fitness profiles
data = df_filtered.iloc[:, dendro['leaves']]    # Order as in dendrogram

# Create line plots
num_genes = data.shape[1]
fig_lines, ax_lines = plt.subplots(num_genes, 1, figsize=(11, 12))
data = data.T   # Make genes be rows instead of columns


for i, (idx, row) in enumerate(data.iterrows()):
    ax_lines[num_genes - 1 - i].plot(row.values, label=idx)
    ax_lines[num_genes - 1 - i].set_xticks([])
    ax_lines[num_genes - 1 - i].set_yticks([])
    if i == 0:
        ax_lines[num_genes - 1 - i].set_xlabel('Cell lines')

plt.show()
