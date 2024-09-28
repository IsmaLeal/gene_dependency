from functions import *

mptn = map_pathway_to_nodes()

hypergraphs = prep_hypergraphs(mptn)

metrics = obtain_hypergraph_metrics(hypergraphs)

metrics.to_csv('results/hg_metrics_final.csv', index=False)