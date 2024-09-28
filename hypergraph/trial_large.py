from functions import prep_graph_networkx, nx_to_gt, edge_density
import pandas as pd
from tqdm import tqdm
from graph_tool.inference import CliqueState
import csv
import numpy as np
import time
import networkx as nx
from sklearn.cluster import SpectralClustering

np.random.seed(143)

g = prep_graph_networkx(0.2, False)

which = 'big'

# Load subgraph
nodes = pd.read_csv(f'hyperedges/unmapped_bigcluster_{which}subcluster_nodes.csv', header=None).T[0].values
subgraph = g.subgraph(nodes)

# Load cluster labels
labels = pd.read_csv(f'hyperedges/unmapped_bigcluster_{which}subcluster_labels.csv', header=None).T[0].values

for i, node in enumerate(subgraph.nodes()):
    subgraph.nodes[node]['label'] = labels[i]

unique_labels = np.unique(labels)
i = 0
for label in unique_labels:
    if i != 2:
        i += 1
        continue
    # Find nodes in this cluster
    nodes_in_cluster = [node for node, data in subgraph.nodes(data=True) if data['label'] == label]
    subsubgraph = subgraph.subgraph(nodes_in_cluster).copy()
    print('Cluster created')

    A = nx.adjacency_matrix(subsubgraph).toarray()
    sc = SpectralClustering(n_clusters=3, affinity='precomputed')
    labels2 = sc.fit_predict(A)

    for j, node in enumerate(subsubgraph.nodes()):
        subsubgraph.nodes[node]['label'] = labels2[j]

    unique_labels2 = np.unique(labels2)
    k = 0
    for label2 in tqdm(unique_labels2):
        nodess = [node for node, data in subsubgraph.nodes(data=True) if data['label'] == label2]
        subsubsubgraph = subsubgraph.subgraph(nodess).copy()

        # Convert to graph-tool
        subsubsubgraph = nx_to_gt(subsubsubgraph)
        print('Converted to graph-tool')

        print(f'This is {k}: {subsubsubgraph.num_vertices()} nodes,'
              f'{subsubsubgraph.num_edges()} edges, and '
              f'density of {edge_density(subsubsubgraph)}')

        if True:
            continue

        # Initialise minimum entropy and best state
        min_entropy = float('inf')
        best_state = None

        # Initialise maximal cliques
        print('Starting CliqueState')
        ts1 = time.time()
        state = CliqueState(subsubsubgraph)
        ts2 = time.time()
        time_cliques = ts2 - ts1
        print(f'\t Done in {time_cliques}s')

        # Perform burn-in sweeps if possible
        if subsubsubgraph.num_vertices() < 500:
            tm11 = time.time()
            print('Starting burn-in period')
            state.mcmc_sweep(niter=5000, beta=1)
            tm12 = time.time()
            print(f'\t Done in {tm12 - tm11}s')
            n = 1000
        elif subsubsubgraph.num_vertices() < 900:
            tm12 = time.time()
            print('Skipping burn-in period')
            n = 500
        else:
            tm12 = time.time()
            print('Skipping burn-in period')
            n = 200

        # Explore the distribution and select minimum-entropy state
        print('Starting posterior distribution exploration')
        for j in tqdm(range(n)):
            state.mcmc_sweep(niter=1, beta=1)
            current_entropy = state.entropy()
            if current_entropy < min_entropy:
                min_entropy = current_entropy
                best_state = state.copy()
        tm21 = time.time()
        print(f'Got those hyperedges in {tm21 - tm12}s! Storing them...')

        # Save hyperedges
        print('Entropy achieved: ', min_entropy)
        hyperedges = []
        for v in best_state.f.vertices():
            if best_state.is_fac[v]:
                continue
            if best_state.x[v] > 0:
                hyperedges.append(best_state.c[v])

        # Express hyperedges in terms of nodes from original graph g
        hyperedges = [tuple(hyperedge) for hyperedge in hyperedges]
        og_hyperedges = []
        for hyperedge in hyperedges:
            new_hyperedge = [subsubsubgraph.vp['node_label'][node] for node in hyperedge]
            og_hyperedges.append(tuple(new_hyperedge))
        tm2 = time.time()
        print(f'Stored in {tm2 - tm21}s.\n Cluster number {k} done')

        # Store hyperedges
        with open(f'hyperedges/unmapped_bigcluster_bigsubcluster_2_{k}.csv', mode='w', newline='') as file:
            writer = csv.writer(file)
            for hyperedge in og_hyperedges:
                writer.writerow(hyperedge)

        k += 1
