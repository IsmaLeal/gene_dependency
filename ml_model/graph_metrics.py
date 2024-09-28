from functions import *
import pandas as pd

# Load graph and dataframe pathway to nodes mapping
g = prep_graph(0.2, False)
mptn = map_pathway_to_nodes()
mptn = mptn.sort_values(by=['# genes'], ascending=True).reset_index().iloc[:, 1:]

graph_metrics = obtain_graph_metrics(g, mptn)
graph_metrics.to_csv('results/g_metrics_final.csv', index=False)