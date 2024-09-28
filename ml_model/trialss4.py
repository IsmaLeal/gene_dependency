import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import networkx as nx
from tqdm import tqdm
import graph_tool.all as gt

options = ['heatmap', 'disagreements', 'networkwtf', 'networkbutgt']
real_option = options[3]

names = pd.read_csv('results/topological_metrics/genes_hg.csv', header=None)[0].values
pathways = pd.read_csv('results/topological_metrics/pathways_hg.csv', header=None)[0].values
y_pred = pd.read_csv('results/topological_metrics/y_pred_topol_hg.csv', header=None)[0].values
y = pd.read_csv('results/topological_metrics/y_topol_hg.csv', header=None)[0].values

names_2 = pd.read_csv('../datasets/names.txt', header=None)[0]
idx_to_name = {idx: name for idx, name in enumerate(names_2)}
name_to_idx = {name: idx for idx, name in enumerate(names_2)}

actualnames = [idx_to_name[index] for index in names]

# Create a DataFrame with the data
data = pd.DataFrame({
    'Gene': names,
    'Gene name': actualnames,
    'Pathway': pathways,
    'Prediction': y_pred,
    'Label': y
})

# Create the 'Disagreement' column, which indicates whether the prediction disagrees with the label
data['Disagreement'] = data['Prediction'] != data['Label']

plt.rc('text', usetex=True)
plt.rc('font', family='serif')
plt.rcParams.update({'font.size': 21})

if real_option == 'heatmap':
    disagreement_matrix = data.pivot_table(index='Gene', columns='Pathway', values='Disagreement', aggfunc='sum',
                                           fill_value=0)

    plt.figure(figsize=(14, 10))
    sns.heatmap(disagreement_matrix, cmap='coolwarm', linewidths=.5)
    plt.title('Heatmap of Prediction Disagreements Across Genes and Pathways')
    plt.show()

if real_option == 'disagreements':
    gene_disagreement_counts = data.groupby('Gene name')['Disagreement'].sum().sort_values(ascending=False)

    plt.figure(figsize=(9, 7))
    gene_disagreement_counts.head(20).plot(kind='bar')
    plt.title('Top 20 genes by number of prediction disagreements')
    plt.ylabel('Number of Disagreements', fontsize=21)
    plt.xticks(rotation=45, fontsize=19)
    plt.grid(True)
    plt.tight_layout()
    plt.show()

if real_option == 'networkwtf':
    # Create a network graph
    G = nx.Graph()

    # Add nodes and edges (Gene to Pathway connections)
    for gene, pathway, pred, label in tqdm(zip(names, pathways, y_pred, y)):
        G.add_node(gene, type='gene')
        G.add_node(pathway, type='pathway')
        G.add_edge(gene, pathway, correct=(pred == label))
    print('checkpoint')
    # Draw the graph with color coding for correct/incorrect predictions
    pos = nx.spring_layout(G)
    print('pos done')
    edge_colors = ['green' if d['correct'] else 'red' for u, v, d in G.edges(data=True)]
    print('drawing')
    nx.draw(G, pos, with_labels=True, edge_color=edge_colors, node_size=500, font_size=10)
    print('drawn')
    plt.title('Gene-Pathway Interaction Network')
    plt.show()

if real_option == 'networkbutgt':
    # Create a Graph object
    g = gt.Graph(directed=False)

    # Add properties to store labels and prediction correctness
    v_prop_name = g.new_vertex_property("string")
    v_prop_type = g.new_vertex_property("string")
    e_prop_correct = g.new_edge_property("bool")
    print(len(names))

    # Create a dictionary to store vertices
    vertex_dict = {}

    # Add nodes (genes and pathways) and edges (gene-pathway interactions)
    for gene, pathway, pred, label in tqdm(zip(names, pathways, y_pred, y)):
        if gene not in vertex_dict:
            v_gene = g.add_vertex()
            vertex_dict[gene] = v_gene
            v_prop_name[v_gene] = gene
            v_prop_type[v_gene] = 'gene'
        else:
            v_gene = vertex_dict[gene]

        if pathway not in vertex_dict:
            v_pathway = g.add_vertex()
            vertex_dict[pathway] = v_pathway
            v_prop_name[v_pathway] = pathway
            v_prop_type[v_pathway] = 'pathway'
        else:
            v_pathway = vertex_dict[pathway]

        edge = g.add_edge(v_gene, v_pathway)
        e_prop_correct[edge] = (pred == label)

    # Set properties
    g.vertex_properties["name"] = v_prop_name
    g.vertex_properties["type"] = v_prop_type
    g.edge_properties["correct"] = e_prop_correct

    gene_color = [0, 0, 1, 1]  # Solid blue (RGBA: no transparency)
    pathway_color = [1, 0.5, 0, 1]  # Solid orange (RGBA: no transparency)
    correct_edge_color = [0, 1, 0, 1]  # RGBA for correct predictions, green
    incorrect_edge_color = [1, 0, 0, 1]  # RGBA for incorrect predictions, red
    # Prepare vertex colors
    vertex_colors = g.new_vertex_property("vector<float>")
    for v in tqdm(g.vertices()):
        if v_prop_type[v] == 'gene':
            vertex_colors[v] = gene_color
        else:
            vertex_colors[v] = pathway_color

    # Prepare edge colors
    edge_colors = g.new_edge_property("vector<float>")
    for e in tqdm(g.edges()):
        if e_prop_correct[e]:
            edge_colors[e] = correct_edge_color
        else:
            edge_colors[e] = incorrect_edge_color

    print('properties done, now sfdp layout')

    gene_disagreement_counts = data.groupby('Gene name')['Disagreement'].sum().sort_values(ascending=False)
    top_20_names = gene_disagreement_counts.head(10).index.tolist()
    top_20 = [name_to_idx[gene] for gene in top_20_names]

    subgraph = gt.Graph(directed=False)

    # Add properties to store labels and prediction correctness
    sub_v_prop_name = subgraph.new_vertex_property("string")
    sub_v_prop_type = subgraph.new_vertex_property("string")
    sub_e_prop_correct = subgraph.new_edge_property("bool")

    # Create a dictionary to store vertices in the subgraph
    sub_vertex_dict = {}

    # Loop through the original graph and add vertices and edges to the subgraph
    pathways_20 = []
    for gene, pathway, pred, label in tqdm(zip(names, pathways, y_pred, y)):
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
    pos = gt.sfdp_layout(subgraph)
    print('which is done!')

    # Add labels to the nodes
    gene_pathway_labels = subgraph.new_vertex_property("string")

    # Set the label for each gene and pathway
    for v in subgraph.vertices():
        v_name = subgraph.vertex_properties["name"][v]
        if v_name in top_20_names:
            gene_pathway_labels[v] = v_name
        elif v_name in pathways_20:
            gene_pathway_labels[v] = v_name
        else:
            gene_pathway_labels[v] = ""  # No label for non-top genes or pathways

    print('now drawing and saving!')
    gt.graph_draw(subgraph, pos,
                  vertex_fill_color=sub_vertex_colors,
                  vertex_text=gene_pathway_labels,vertex_text_position=1,  # Adjust position if necessary
                  vertex_font_size=6,  # Adjust font size for readability
                  vertex_text_color="black",  # Set text color to black
                  vertex_size=3,
                  edge_color=sub_edge_colors,
                  output_size=(900, 900), output='results/embeddings/besthypergraph.png')
    gt.graph_draw(subgraph, pos,
                  vertex_fill_color=sub_vertex_colors,
                  vertex_text=gene_pathway_labels,vertex_text_position=1,  # Adjust position if necessary
                  vertex_font_size=6,  # Adjust font size for readability
                  vertex_text_color="black",  # Set text color to black
                  vertex_size=3,
                  edge_color=sub_edge_colors,
                  output_size=(900, 900), output='results/embeddings/besthypergraph.pdf')