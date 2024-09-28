import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
import graph_tool.all as gt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import accuracy_score


def check_class_balance(y):
    unique, counts = np.unique(y, return_counts=True)
    balance = dict(zip(unique, counts))
    print("Class balance:", balance)


def get_class_weights(y):
    classes = np.unique(y)
    class_weights = compute_class_weight('balanced', classes=classes, y=y)
    return dict(zip(classes, class_weights))


metrics = pd.read_csv('results/topological_metrics/hg_topological_metrics.csv')

# Load CRISPR essentiality labels
labels = pd.read_csv('../datasets/CRISPRInferredCommonEssentials.csv')

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
names = pd.read_csv('results/topological_metrics/genes_hg.csv', header=None)[0].values
pathways = pd.read_csv('results/topological_metrics/pathways_hg.csv', header=None)[0].values

names_2 = pd.read_csv('../datasets/names.txt', header=None)[0]
idx_to_name = {idx: name for idx, name in enumerate(names_2)}
name_to_idx = {name: idx for idx, name in enumerate(names_2)}

actualnames = [idx_to_name[index] for index in names]

# Train the Random Forest Classifier
print('Training model')
rf = RandomForestClassifier(n_estimators=500, class_weight=class_dict, random_state=42)
rf.fit(X_train, y_train)

# Predict the class labels
print('Predicting labels')
y_pred = rf.predict(X_test)
print(accuracy_score(y_test, y_pred))

# Create a DataFrame with the data
data = pd.DataFrame({
    'Gene': names,
    'Gene name': actualnames,
    'Pathway': pathways,
    'Prediction': y_pred,
    'Label': y_test
})

# Predict the class probabilities
print('Getting probabilities')
probs = rf.predict_proba(X_test)

# Get indices of all gene-pathway pairs with 100% reliability
indices = np.zeros(len(X_test))
for idx, i in enumerate(probs[:, 0]):
    if i >= 0.5 or i <= 0.55:
        indices[idx] = 1
for idx, i in enumerate(probs[:, 1]):
    if i >= 0.5 or i <= 0.55:
        indices[idx] = 1

data['Low confidence'] = indices

cosa = data.groupby('Gene name')['Low confidence'].sum().sort_values(ascending=False)
top_20_names = cosa.head(20).index.tolist()
top_20 = [name_to_idx[gene] for gene in top_20_names]

plt.rc('text', usetex=True)
plt.rc('font', family='serif')
plt.rcParams.update({'font.size': 21})

subgraph = gt.Graph(directed=False)

# Add properties to store labels and prediction correctness
sub_v_prop_name = subgraph.new_vertex_property("string")
sub_v_prop_type = subgraph.new_vertex_property("string")
sub_e_prop_correct = subgraph.new_edge_property("bool")

# Create a dictionary to store vertices in the subgraph
sub_vertex_dict = {}

# Loop through the original graph and add vertices and edges to the subgraph
pathways_20 = []
for gene, pathway, pred, label in tqdm(zip(names, pathways, y_pred, y_test)):
    if gene in top_20:
        pathways_20.append(pathway)
        if gene not in sub_vertex_dict:
            v_gene = subgraph.add_vertex()
            sub_vertex_dict[gene] = v_gene
            sub_v_prop_name[v_gene] = gene
            sub_v_prop_type[v_gene] = 'gene'
        else:
            v_gene = sub_vertex_dict[gene]

        if pathway not in sub_vertex_dict:
            v_pathway = subgraph.add_vertex()
            sub_vertex_dict[pathway] = v_pathway
            sub_v_prop_name[v_pathway] = pathway
            sub_v_prop_type[v_pathway] = 'pathway'
        else:
            v_pathway = sub_vertex_dict[pathway]

        edge = subgraph.add_edge(v_gene, v_pathway)
        sub_e_prop_correct[edge] = (pred == label)

# Set properties for the subgraph
subgraph.vertex_properties["name"] = sub_v_prop_name
subgraph.vertex_properties["type"] = sub_v_prop_type
subgraph.edge_properties["correct"] = sub_e_prop_correct

# Assign colors as before
gene_color = [0, 0, 1, 1]  # Solid blue (RGBA: no transparency)
pathway_color = [1, 0.5, 0, 1]  # Solid orange (RGBA: no transparency)
correct_edge_color = [0, 1, 0, 1]  # RGBA for correct predictions, green
incorrect_edge_color = [1, 0, 0, 1]  # RGBA for incorrect predictions, red

# Prepare vertex colors
sub_vertex_colors = subgraph.new_vertex_property("vector<float>")
for v in tqdm(subgraph.vertices()):
    if sub_v_prop_type[v] == 'gene':
        sub_vertex_colors[v] = gene_color
    else:
        sub_vertex_colors[v] = pathway_color

# Prepare edge colors
sub_edge_colors = subgraph.new_edge_property("vector<float>")
for e in tqdm(subgraph.edges()):
    if sub_e_prop_correct[e]:
        sub_edge_colors[e] = correct_edge_color
    else:
        sub_edge_colors[e] = incorrect_edge_color

# Visualize the network
pos = gt.sfdp_layout(subgraph, C=2)
for v in subgraph.vertices():
    pos[v][0] *= 10
    pos[v][1] *= 10
print('which is done!')

# Add labels to the nodes
gene_pathway_labels = subgraph.new_vertex_property("string")

# Set the label for each gene and pathway
for v in subgraph.vertices():
    v_name = subgraph.vertex_properties["name"][v]
    try:
        if int(float(v_name)) in top_20:
            gene_pathway_labels[v] = idx_to_name[int(float(v_name))]
        elif v_name in pathways_20:
            gene_pathway_labels[v] = ''
        else:
            gene_pathway_labels[v] = ''  # No label for non-top genes or pathways
    except:
        if v_name in top_20:
            gene_pathway_labels[v] = v_name
        elif v_name in pathways_20:
            gene_pathway_labels[v] = ''
        else:
            gene_pathway_labels[v] = ''  # No label for non-top genes or pathways

print('now drawing and saving!')
gt.graph_draw(subgraph, pos,
              vertex_fill_color=sub_vertex_colors,
              vertex_text=gene_pathway_labels,vertex_text_position=1,  # Adjust position if necessary
              vertex_font_size=6,  # Adjust font size for readability
              vertex_text_color="black",  # Set text color to black
              vertex_text_background='white',
              vertex_size=6,
              edge_color=sub_edge_colors,
              output_size=(600, 600), output='results/aquihostia.png')
gt.graph_draw(subgraph, pos,
              vertex_fill_color=sub_vertex_colors,
              vertex_text=gene_pathway_labels,vertex_text_position=1,  # Adjust position if necessary
              vertex_font_size=6,  # Adjust font size for readability
              vertex_text_color='black',  # Set text color to black
              vertex_text_background='white',
              vertex_size=6,
              edge_color=sub_edge_colors,
              output_size=(600, 600), output='results/aquihostia.pdf')


