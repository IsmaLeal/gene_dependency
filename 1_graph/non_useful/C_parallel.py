import os
import graph_tool.all as gt
from scipy.special import comb
import pandas as pd
import numpy as np
import time
from tqdm import tqdm
from multiprocessing import Pool, Manager

def clean_col_names(col):
    return col.split(' (')[0]


def init_worker():
    seed = os.getpid()
    np.random.seed(seed)


def prep_graph(threshold):
    gt.openmp_set_num_threads(4)

    print('Preparing graph...')
    start_time = time.time()
    corrs = pd.read_csv('rank_transf_symm_2.csv', delimiter=',', index_col=0)
    corrs /= np.max(corrs.values)
    gene_names = np.array([clean_col_names(col) for col in corrs.columns])
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


def single_rewiring(internal_nodes, external_nodes, degrees, progress_list):
    current_degrees = degrees.copy()

    # stubs will be a list with internal node indices, where each node will appear as many times as degree it has
    stubs = []
    for node in internal_nodes:
        stubs.extend([node] * degrees[node])
    np.random.shuffle(stubs)

    # Initialise expected values
    E_int = 0
    E_ext = 0

    while len(stubs) > 1:
        # Select a node
        node = stubs.pop()

        # Compute the degrees needed to obtain the probability of the edge being internal
        D_int = sum(current_degrees[i] for i in internal_nodes if i != node)
        D_tot = sum(current_degrees[i] for i in internal_nodes + external_nodes if i != node)
        if D_tot == 0:
            continue
        p_i = D_int / D_tot

        # Update the expected values (probabilities weighted by 1 (the amount of edges associated with that probability)
        E_int += p_i
        E_ext += 1 - p_i

        # Choose internal or external edge based on the probability
        if np.random.random() < p_i:
            targets = [i for i in internal_nodes if i != node and i in stubs]
            target = np.random.choice(targets)
            stubs.remove(target)
        else:
            target = np.random.choice(external_nodes)

        current_degrees[node] -= 1
        current_degrees[target] -= 1

    # Calculate EDR for this iteration
    possible_internal_edges = comb(len(internal_nodes), 2)
    possible_external_edges = len(internal_nodes) * len(external_nodes)

    ied = E_int / possible_internal_edges
    eed = E_ext / possible_external_edges
    edr = ied / eed if eed != 0 else float('inf')

    progress_list.append(1)
    return edr


def simulate_rewiring(g, internal_nodes, num_iterations=1000):
    '''
    :param g: graph-tool graph object
    :param internal_vertices: list of vertex indices that are internal
    :param num_iterations:
    :return:
    '''
    n_total = g.num_vertices()
    set_internal = set(internal_nodes)
    set_external = set(range(n_total)) - set_internal
    external_nodes = list(set_external)

    # Keep track of original degree distribution of internal nodes
    degrees = {int(v): v.out_degree() for v in g.vertices()}

    with Manager() as manager:
        progress_list = manager.list()
        with Pool(processes=32, initializer=init_worker) as pool:
            results = [pool.apply_async(single_rewiring, args=(internal_nodes, external_nodes, degrees, progress_list)) for _ in range(num_iterations)]

            # Use tqdm to monitor progress
            with tqdm(total=num_iterations) as pbar:
                while len(progress_list) < num_iterations:
                    pbar.update(len(progress_list) - pbar.n)
                    pbar.refresh()
                    #time.sleep(0.1)

            # Get results from all iterations
            ratios = [result.get() for result in results]

    # Create masks needed to compute observed EDR
    internal_mask = g.new_vertex_property('bool')
    external_mask = g.new_vertex_property('bool')
    for v in g.vertices():
        idx = int(v)
        internal_mask[v] = idx in internal_nodes
        external_mask[v] = idx in list(set_external)

    # Get observed EDR
    observed = edge_density_ratio(g, internal_mask, external_mask)

    p_value = np.mean([r >= observed for r in ratios])
    return observed, p_value, ratios


if __name__ == '__main__':
    g, gene_names = prep_graph(0.8)
    internal_vertices = []




