from graph_tool.all import Graph, minimize_blockmodel_dl, openmp_set_num_threads
from functions import prep_graph
import pickle

openmp_set_num_threads(30)
# # Perform the community detection (fitting a stochastic block model)
# # by minimising its description length using agglomerative heuristic
# g, _ = prep_graph(0.9)
# state = minimize_blockmodel_dl(g)   # ~3h
#
# # Create a subgraph for each community or block: ~6h
# membership = state.get_blocks() # Vertex property map assigning a community index to each node
# subgraphs = []
# for block_id in set(membership.a):
#     nodes = [v for v in g.vertices() if membership[v] == block_id]  # Isolate nodes belonging to the block
#     subgraph = Graph(directed=False)
#     node_map = {}   # Dict. mapping each general vertex v to the node new_v from the subgraph
#     for v in nodes:
#         new_v = subgraph.add_vertex()
#         node_map[int(v)] = int(new_v)
#     subgraph_edges = [(node_map[int(e.source())], node_map[int(e.target())]) for e in g.edges() if
#                       membership[e.source()] == block_id and membership[e.target()] == block_id]
#     subgraph.add_edge_list(subgraph_edges)
#     subgraphs.append(subgraph)
#
# # Store subgraphs for later use
# for i, subgraph in enumerate(subgraphs):
#     f = f'subgraphs/subgraph_{i}.gt'
#     subgraph.save(f)
#
# # Store membership of each node
# membership_list = list(membership.a)
# with open('membership.pkl', 'wb') as f:
#     pickle.dump(membership_list, f)


def load_membership(g):
    with open('membership.pkl', 'rb') as f:
        membership_list = pickle.load(f)
    membership = g.new_vertex_property('int')
    for i, value in enumerate(membership_list):
        membership[g.vertex(i)] = value

    return membership
