from functions import *

# Load graph and dataframe pathway to nodes mapping
mptn = map_pathway_to_nodes()
print('mptn done, going for graph')
g = prep_graph(0.2, False)
mptn = mptn.sort_values(by=['# genes'], ascending=True).reset_index().iloc[:, 1:]

graph_embeddings = obtain_graph_embeddings(g, mptn, dim=128)
graph_embeddings.to_csv('results/embeddings/g_embeddings_128d.csv', index=False)