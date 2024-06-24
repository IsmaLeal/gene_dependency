import graph_tool.all as gt
import hypernetx as hnx
import matplotlib.pyplot as plt
from graph_tool.draw import graph_draw
from PIL import Image

names = ['a', 'b', 'c', 'd', 'e', 'f']

hyperedges = {
    'e1': ['a', 'b', 'c'],
    'e2': ['b', 'c', 'd', 'e'],
    'e3': ['c', 'd'],
    'e4': ['f', 'e']
}

edges = [
    [0, 1],
    [0, 2],
    [1, 2],
    [1, 3],
    [1, 4],
    [2, 3],
    [2, 4],
    [3, 4],
    [4, 5]
]

g = gt.Graph(directed=False)
g.add_vertex(n=len(names))
g.add_edge_list(edges)
labels = g.new_vertex_property('string', vals=names)

pos = gt.sfdp_layout(g)

rotated_pos = g.new_vertex_property("vector<double>")
for v in g.vertices():
    x, y = pos[v]
    rotated_pos[v] = (y, -x)

vertex_fill_color = g.new_vertex_property("vector<double>")
for v in g.vertices():
    vertex_fill_color[v] = [0, 0, 0, 1]

graph_draw(
    g,
    pos,
    output='here.pdf',
    vertex_text=labels,
    vertex_size=15,
    vertex_fill_color=vertex_fill_color,
    output_size=(700, 300))

#H = hnx.Hypergraph(hyperedges)

#fig, ax = plt.subplots(1, 1, figsize=(3, 3))
#hnx.draw(H, with_node_labels=True, with_edge_labels=False, ax=ax)
#plt.show()
