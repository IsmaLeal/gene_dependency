from functions import prep_graph_networkx, spectral_clustering_load_labels, edge_density
import graph_tool.all as gt
from graph_tool.inference import CliqueState
import time
import csv
import pandas as pd

def nx_to_gt(graph):
	gt_graph = gt.Graph(directed=False)
	gt_graph.vertex_properties["node_label"] = gt_graph.new_vertex_property("int")
	node_map = {}

	# Add nodes
	for node in graph.nodes():
		v = gt_graph.add_vertex()
		node_map[node] = v
		gt_graph.vp.node_label[v] = node

	# Add edges
	for edge in graph.edges(data=True):
		u, v, data = edge
		gt_graph.add_edge(node_map[u], node_map[v])

	return gt_graph


g = prep_graph_networkx(0.2, False)
print('Creating clusters!')
clusters_nx, mptn = spectral_clustering_load_labels(g)
print('Clusters returned!')

clusters_dict = {k: nx_to_gt(cluster) for k, cluster in clusters_nx.items()}
print('Dictionary done!')
clusters = [v for k, v in clusters_dict.items()]
#clusters = clusters[::-1]

times_c = []
times_s = []
all_hyperedges = []
entropies = []
cluster_numbers = ['Large0', 'Large1', 'Large2', 'Small']

i = 0
for subgraph in clusters:
	print(f'This is: cluster {cluster_numbers[i]},'
		  f'with {subgraph.num_vertices()} nodes,'
		  f'{subgraph.num_edges()} edges, and'
		  f'density of {edge_density(subgraph)}')
	min_entropy = float('inf')
	best_state = None

	print('Starting CliqueState')
	ts1 = time.time()
	state = CliqueState(subgraph)
	ts2 = time.time()
	time_cliques = ts2 - ts1
	print(f'\t Done in {time_cliques}s')
	times_c.append(time_cliques)

	tm11 = time.time()
	print('Starting burn-in period')
	state.mcmc_sweep(niter=10000, beta=1)
	tm12 = time.time()
	print(f'\t Done in {tm12 - tm11}s')

	print('Starting posterior distribution exploration')
	for i in range(1000):
		state.mcmc_sweep(niter=1, beta=1)
		current_entropy = state.entropy()
		if current_entropy < min_entropy:
			min_entropy = current_entropy
			best_state = state.copy()
	tm21 = time.time()
	print(f'Got those hyperedges in {tm21 - tm12}s! Storing them...')
	times_s.append(tm21 - tm11)

	entropies.append(best_state.entropy())
	hyperedges = []
	for v in best_state.f.vertices():
		if best_state.is_fac[v]:
			continue
		if best_state.x[v] > 0:
			hyperedges.append(best_state.c[v])

	hyperedges = [tuple(hyperedge) for hyperedge in hyperedges]
	tm2 = time.time()
	print(f'Stored in {tm2-tm21}s.\n Cluster number {i} done')

	with open(f'hyperedges/bigbro{i}.csv', mode='w', newline='') as file:
		writer = csv.writer(file)
		for hyperedge in hyperedges:
			writer.writerow(hyperedge)

	all_hyperedges.append(hyperedges)
	i += 1

with open('WHAT.csv', mode='w', newline='') as file:
	writer = csv.writer(file)

	for sublist in all_hyperedges:
		for item in sublist:
			writer.writerow(item)

df = pd.DataFrame({'Cluster': cluster_numbers,
				   'Hyperedges': all_hyperedges,
				   'Entropy': entropies,
				   'Time cliques': times_c,
				   'Time sweeps': times_s})

df.to_csv('THISISGREAT.csv', index=False)
