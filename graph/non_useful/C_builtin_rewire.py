from tqdm import tqdm
from non_useful.B_adjacency_matrix import create_adjacency_matrix
import numpy as np
import graph_tool.all as gt
import time


def clean_col_names(col):
    return col.split(' (')[0]


def indices_dict(genes, names):
    return {name: np.where(genes == name)[0].tolist() for name in names}


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


def p_values(A, labels, names):
    gt.openmp_set_num_threads(30)

    # Create a mapping from labels to indices
    label_to_index = {label: idx for idx, label in enumerate(labels)}

    # Create the graph-tool graph
    g = gt.Graph(directed=False)
    g.add_vertex(n=A.shape[0])                                     # Add vertices
    g.add_edge_list(np.transpose(np.nonzero(np.triu(A, 1))))    # Add edges

    # Create sets of internal & external nodes
    internal_indices = set(label_to_index[name] for name in names)
    external_indices = set(range(A.shape[0])) - internal_indices

    # Create masks for internal & external nodes
    internal_mask = g.new_vertex_property('bool')
    external_mask = g.new_vertex_property('bool')
    for v in tqdm(g.vertices()):
        idx = int(v)
        internal_mask[v] = idx in internal_indices
        external_mask[v] = idx in external_indices

    observed_ratio = edge_density_ratio(g, internal_mask, external_mask)
    print('observed ratio obtained')

    ratios = []
    for _ in tqdm(range(200)):
        print('\nRewiring started...')
        start_rewiring = time.time()
        gt.random_rewire(g, model='configuration', edge_sweep=1)
        end_rewiring = time.time()
        times_rewiring = end_rewiring - start_rewiring
        print(f'Rewiring complete: {int(times_rewiring/60)}min {int(times_rewiring%60)}s')

        print('Calculating EDR...')
        start_edr = time.time()
        ratio = edge_density_ratio(g, internal_mask, external_mask)
        end_edr = time.time()
        times_edr = end_edr - start_edr
        print(f'EDR obtained in: {int(times_edr/60)}min {int(times_edr%60)}s')

        ratios.append(ratio)

    p_value = np.mean([r >= observed_ratio] for r in ratios)
    return observed_ratio, p_value

    # #graph_draw(g, output='graph.png')


if __name__ == '__main__':
    A = create_adjacency_matrix(0.5)
    print('Adjacency matrix created')
    # Find gene names
    new_col_names = np.array([clean_col_names(col) for col in A.columns])

    # Specify names of protein complex of interest
    names = np.array(['ADD2', 'DMTN', 'SLC2A1'])  # , 'GLUT1', 'ACAP1', 'CLTC', 'SLC2A4', 'GLUT4'])





