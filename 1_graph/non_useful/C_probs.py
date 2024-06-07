import graph_tool.all as gt
from scipy.special import comb
import pandas as pd
import numpy as np
import time
from tqdm import tqdm


def clean_col_names(col):
    return col.split(' (')[0]


def prep_graph(threshold):
    print('Preparing graph...')
    start_time = time.time()
    corrs = pd.read_csv('../rank_transf_symm_2.csv', delimiter=',', index_col=0)
    corrs /= np.max(corrs.values)
    gene_names = np.array([clean_col_names(col) for col in corrs.columns])
    threshold = threshold
    A = (corrs.values > threshold).astype(int)

    g = gt.Graph(directed=False)
    g.add_vertex(n=A.shape[0])
    edges = np.transpose(np.nonzero(np.triu(A, 1)))
    g.add_edge_list(edges)
    print(f'Data loaded in {time.time() - start_time:.2} seconds.')

    return g, gene_names


def count_external_edges(g, internal_mask, external_mask):
    internal_view = gt.GraphView(g, vfilt=internal_mask)
    external_view = gt.GraphView(g, vfilt=external_mask)

    return g.num_edges() - internal_view.num_edges() - external_view.num_edges()


def edge_density_ratio(g, internal_mask, external_mask):
    subgraph = gt.GraphView(g, vfilt=internal_mask)
    n_internal_edges = subgraph.num_edges()
    possible_internal_edges = sum(internal_mask.get_array()) * (sum(internal_mask.get_array()) - 1) / 2
    ied = n_internal_edges / possible_internal_edges

    n_external_edges = count_external_edges(g, internal_mask, external_mask)
    possible_external_edges = sum(internal_mask.get_array()) * sum(external_mask.get_array())
    eed = n_external_edges / possible_external_edges

    return ied / eed if eed != 0 else float('inf')


def simulate_rewiring(g, internal_nodes, num_iterations=200):
    '''
    :param g: graph-tool graph object
    :param internal_vertices: list of vertex indices that are internal
    :param num_iterations:
    :return:
    '''
    gt.openmp_set_num_threads(60)
    # 30 threads -> 200s per iteration
    # 60 threads -> 196s per iteration
    # 80 threads -> 170s per iteration

    n_total = g.num_vertices()
    set_internal = set(internal_nodes)
    set_external = set(range(n_total)) - set_internal
    external_nodes = list(set_external)

    # Dictionary to store vertex degrees
    degrees = {v: v.out_degree() for v in g.vertices()}

    ratios = []

    for _ in tqdm(range(num_iterations)):
        # Initialise edge counts
        internal_edges = 0
        external_edges = 0

        # Work with a copy of the degrees to modify during simulation
        current_degrees = degrees.copy()

        # Recalculate probabilities and rewire the network
        for i in internal_nodes:         # For each node
            # Number of outgoing edges
            n_edges = degrees[i]
            for _ in range(n_edges): # and each edge
                # Skip if no more connections to make
                if current_degrees[i] <= 0:
                    continue

                # Obtain current degree sums (internal & total)
                internal_degree_sum = sum(current_degrees[j] for j in internal_nodes if j != i)
                total_degree_sum = sum(current_degrees[k] for k in internal_nodes + external_nodes if k != i)

                # Prevent division by 0
                if total_degree_sum == 0:
                    continue

                # Obtain probability that edge under consideration is internal
                p_internal = internal_degree_sum / total_degree_sum

                # Choose internal or external
                if np.random.random() < p_internal:
                    # Available internal node choices
                    choices = [j for j in internal_nodes if j != i and current_degrees[j] > 0]
                    # Skip if no more connections to make
                    if not choices:
                        continue
                    # Randomly choose an internal node
                    j = np.random.choice(choices)
                    internal_edges += 1
                else:
                    # Available external node choices
                    choices = [j for j in external_nodes if current_degrees[j] > 0]
                    # Skip if no more connections to make
                    if not choices:
                        continue
                    # Randomly choose an external node
                    j = np.random.choice(choices)
                    external_edges += 1

                # Decrement degrees
                current_degrees[i] -= 1
                current_degrees[j] -= 1

                a = sum([current_degrees[i] for i in internal_nodes])
                if a % 100 == 0:
                    print(a)

        # Calculate EDR for this iteration
        possible_internal_edges = comb(len(internal_nodes), 2)
        possible_external_edges = len(internal_nodes) * len(external_nodes)

        ied = internal_edges / possible_internal_edges
        eed = external_edges / possible_external_edges
        edr = ied / eed if eed != 0 else float('inf')
        ratios.append(edr)

    internal_mask = g.new_vertex_property('bool')
    external_mask = g.new_vertex_property('bool')

    for v in g.vertices():
        idx = int(v)
        internal_mask[v] = idx in internal_nodes
        external_mask[v] = idx in external_nodes

    observed_edr = edge_density_ratio(g, internal_mask, external_mask)

    p_value = sum(r >= observed_edr for r in ratios) / len(ratios)
    return observed_edr, p_value


if __name__ == '__main__':
    g, gene_names = prep_graph(0.8)
    internal_vertices = []




