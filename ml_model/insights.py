import pandas as pd
from tqdm import tqdm
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import accuracy_score

def filter_CORUM():
    # Load CORUM dataset
    file_path = '../datasets/humanComplexes.txt'
    try:
        df = pd.read_csv(file_path, delimiter='\t')
    except:
        print(f'File {file_path} does not exist. Please re-download from CORUM website')
        df = None

    # Select specific columns
    complexes = df[['ComplexID', 'ComplexName', 'Cell line', 'subunits(Gene name)', 'GO description', 'FunCat description']]

    # Sort by 'Cell line'
    complexes = complexes.sort_values(by=['Cell line'])

    # Exclude 'complexes' including only one subunit/ gene
    mask_mono = [len(complexes['subunits(Gene name)'].values[i].split(';')) > 1 for i in range(len(complexes))]
    complexes = complexes.loc[mask_mono]

    # Return as dataframe
    return complexes


def check_class_balance(y):
    unique, counts = np.unique(y, return_counts=True)
    balance = dict(zip(unique, counts))
    print("Class balance:", balance)


def get_class_weights(y):
    classes = np.unique(y)
    class_weights = compute_class_weight('balanced', classes=classes, y=y)
    return dict(zip(classes, class_weights))


def load_CORUM(filtered=False):
    '''
    Loads the previously created dataset of complexes of interest, and returns the list of names
    '''

    if filtered:
        # Load complexes
        file_path = '../datasets/filtered_complexes.csv'
        try:
            df = pd.read_csv(file_path)
            # Process protein subunit names for each complex:
            # convert from one string per complex in the column 'subunits(Gene name)'
            # to one list per complex in 'complexes'
            complexes_strings = df['subunits(Gene name)'].values
            complexes = [complex.split(';') for complex in complexes_strings]

        except:
            print(f'File {file_path} does not exist. Please run filter_CORUM()')
            complexes = None

    else:
        df = filter_CORUM()
        complexes_strings = df['subunits(Gene name)'].values
        complexes = [complex.split(';') for complex in complexes_strings]

    return complexes


metrics = pd.read_csv('results/topological_metrics/hg_topological_metrics.csv')

# Load CRISPR essentiality labels
labels = pd.read_csv('../datasets/CRISPRInferredCommonEssentials.csv')

names = pd.read_csv('../datasets/names.txt', header=None)[0]
idx_to_name = {idx: name for idx, name in enumerate(names)}
name_to_idx = {name: idx for idx, name in enumerate(names)}

# Merge both
# hypergraph...
metrics['Essential'] = metrics['Gene name'].isin(labels['Essentials']).astype(int)

# Define the features and the target
X = metrics.drop(columns=['Gene', 'Gene name', 'Pathway', 'Essential'])
y = metrics['Essential']

check_class_balance(y)

# Class imbalance
classes = np.unique(y)
class_weights = compute_class_weight('balanced', classes=classes, y=y)
class_dict = dict(zip(classes, class_weights))

# Split data
print('Splitting data')
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the Random Forest Classifier
print('Training model')
rf = RandomForestClassifier(n_estimators=500, class_weight=class_dict, random_state=42)
rf.fit(X_train, y_train)

# Predict the class labels
print('Predicting labels')
y_pred = rf.predict(X_test)
print(accuracy_score(y_test, y_pred))

# Predict the class probabilities
print('Getting probabilities')
probs = rf.predict_proba(X_test)

# Get indices of all gene-pathway pairs with at least 80% reliability
idx1 = probs[:, 0] >= 0.8
idx2 = probs[:, 1] >= 0.8
indices = idx1 + idx2
entry_index = np.array(X_test[indices].index)

complexes = load_CORUM(filtered=False)
complexes_df = filter_CORUM()
hyperedges = pd.read_csv('../hypergraph/hyperedges/hyperedges_final.csv')

n = 0
total = 0
cell_lines = []
complex_indices = []
those_pathways = []
for idx in tqdm(entry_index):
    gene_pathway = metrics.loc[idx]
    gene = gene_pathway['Gene name']
    pathway = gene_pathway['Pathway']
    if pathway == 'U-mid-14':
        continue
    row = hyperedges[hyperedges['Pathway'] == pathway]
    current_hyperedges = eval(row['Hyperedges (gene names)'].values[0])

    for hyperedge in current_hyperedges:
        if gene not in hyperedge:
            continue
        total += 1
        for complex_ in complexes:
            if set(complex_).issubset(set(hyperedge)):
                if gene not in complex_:
                    continue
                n += 1
                weird = ''
                for gene2 in complex_:
                    weird += str(gene2) + ';'
                weird = weird[:-1]
                cell_line_s = complexes_df[complexes_df['subunits(Gene name)'] == weird]['Cell line']
                cell_lines.append(cell_line_s)
                complex_idx = complexes_df[complexes_df['subunits(Gene name)'] == weird]['ComplexID']
                complex_indices.append(complex_idx)
                those_pathways.append(pathway)
                break
        # if hyperedge in complexes:
        #     n += 1
        #     print(pathway, hyperedge)
cell_lines_all = pd.concat(cell_lines).values
indices_all = pd.concat(complex_indices).values
print(np.unique(indices_all))

alll = set([])
for cell_line in cell_lines_all:
    alll.add(cell_line)

# # Count number of hyperedges computed that are a complex
# for idx, row in tqdm(hyperedges.iterrows()):
#     current_hyperedges = eval(row['Hyperedges (gene names)'])
#     for hyperedge in current_hyperedges:
#         total += 1
#         if hyperedge in complexes:
#             n += 1
a = 1
print(f'From {total} hyperedges, found {n} protein complexes')