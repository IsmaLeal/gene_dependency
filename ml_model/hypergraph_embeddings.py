from functions import *

# Load graph and dataframe pathway to nodes mapping
mptn = map_pathway_to_nodes()
mptn = mptn.sort_values(by=['# genes'], ascending=True).reset_index().iloc[:, 1:]

hypergraphs = prep_hypergraphs(mptn)

hypergraph_embeddings = obtain_hypergraph_embeddings(hypergraphs, 'nodes', dim=5)
hypergraph_embeddings.to_csv('results/embeddings/hg_embeddings_5d.csv', index=False)