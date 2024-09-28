from functions import prep_graph_networkx, eigvals, nx_to_gt, edge_density
import pandas as pd
from tqdm import tqdm
import graph_tool.all as gt
from graph_tool.inference import CliqueState
from sklearn.cluster import SpectralClustering
from graph_tool.spectral import adjacency
import csv
import networkx as nx
import numpy as np
import time
import matplotlib.pyplot as plt

np.random.seed(143)

mid_nodes = pd.read_csv('hyperedges/unmapped_middlecluster_nodes.csv', header=None).T[0].values
g = prep_graph_networkx(0.2, False)

print('Creating subgraph')
subgraph = g.subgraph(mid_nodes)
A = nx.adjacency_matrix(subgraph)

print('Spectral clustering')
sc = SpectralClustering(n_clusters=17, affinity='precomputed')
labels = sc.fit_predict(A)

for i, node in enumerate(subgraph.nodes()):
    subgraph.nodes[node]['label'] = labels[i]

times_c = []
times_s = []
all_hyperedges = []
entropies = []

i = 0

unique_labels = np.unique(labels)
for label in tqdm(unique_labels):
    if i != 14:
        i += 1
        continue
    # Find nodes with this label
    nodes_in_cluster = [node for node, data in subgraph.nodes(data=True) if data['label'] == label]
    print(f'Cluster with {len(nodes_in_cluster)} nodes')
    subsubgraph = subgraph.subgraph(nodes_in_cluster).copy()
    print('Cluster created')

    subsubgraph = nx_to_gt(subsubgraph)
    print('Converted to graph-tool')

    print(f'This is {i}: {subsubgraph.num_vertices()} nodes,'
          f'{subsubgraph.num_edges()} edges, and '
          f'density of {edge_density(subsubgraph)}')

    min_entropy = float('inf')
    best_state = None

    print('Starting CliqueState')
    ts1 = time.time()
    state = CliqueState(subsubgraph)
    ts2 = time.time()
    time_cliques = ts2 - ts1
    print(f'\t Done in {time_cliques}s')
    times_c.append(time_cliques)

    tm11 = time.time()
    print('Starting burn-in period')
    #state.mcmc_sweep(niter=10000, beta=1)
    tm12 = time.time()
    print(f'\t Done in {tm12 - tm11}s')

    print('Starting posterior distribution exploration')
    for j in tqdm(range(500)):
        state.mcmc_sweep(niter=1, beta=1)
        current_entropy = state.entropy()
        if current_entropy < min_entropy:
            min_entropy = current_entropy
            best_state = state.copy()
    tm21 = time.time()
    print(f'Got those hyperedges in {tm21 - tm12}s! Storing them...')
    times_s.append(tm21 - tm12)

    entropies.append(best_state.entropy())
    hyperedges = []
    for v in best_state.f.vertices():
        if best_state.is_fac[v]:
            continue
        if best_state.x[v] > 0:
            hyperedges.append(best_state.c[v])

    hyperedges = [tuple(hyperedge) for hyperedge in hyperedges]
    og_hyperedges = []
    for hyperedge in hyperedges:
        new_hyperedge = [subsubgraph.vp['node_label'][node] for node in hyperedge]
        og_hyperedges.append(tuple(new_hyperedge))
    tm2 = time.time()
    print(f'Stored in {tm2 - tm21}s.\n Cluster number {i} done')

    with open(f'hyperedges/unmapped_middlecluster_{i}.csv', mode='w', newline='') as file:
        writer = csv.writer(file)
        for hyperedge in og_hyperedges:
            writer.writerow(hyperedge)

    all_hyperedges.append(hyperedges)
    i += 1

with open(f'hyperedges/unmapped_middlecluster_labels.csv', mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(labels)

a = 1

# eigvals = eigvals(A, k=21)
#
# plt.rc('text', usetex=True)
# plt.rc('font', family='serif')
# plt.rcParams.update({'font.size': 21})
# plt.figure(figsize=(9, 6))
# plt.plot(np.arange(1, 21), eigvals[1:], linestyle='-', marker='o')
# plt.xlabel('Index')
# plt.ylabel('Eigenvalue')
# plt.title('Smallest 20 eigenvalues of medium cluster')
# plt.grid(True)
# plt.tight_layout()
# plt.show()
