import sys
sys.path.append('/home/lealaskerova/diss/conda_env/code/ml_model/HNHN/')
import HNHN.hypergraph as hh
import torch
from tqdm import tqdm
import pandas as pd
from functions import prep_hypergraphs, map_pathway_to_nodes
import argparse

mptn = map_pathway_to_nodes()
hypergraphs = prep_hypergraphs(mptn)

names = pd.read_csv('../datasets/names.txt', header=None)[0]
idx_to_name = {idx: name for idx, name in enumerate(names)}

i = 0
dfs = []
for pathway, h in tqdm(hypergraphs.items()):
    hyperedges = h.get_edges()
    ne = len(hyperedges)

    nodes = h.get_nodes()
    nv = len(nodes)

    print(f'Starting {i} with {nv} nodes and {ne} hyperedges')

    node_to_idx = {node: idx for idx, node in enumerate(nodes)}

    vidx, eidx, paper_author = [], [], []
    for idx, hyperedge in enumerate(hyperedges):
        for node in hyperedge:
            vidx.append(node_to_idx[node])
            eidx.append(idx)
            paper_author.append([node_to_idx[node], idx])

    vidx = torch.tensor(vidx, dtype=torch.int64)
    eidx = torch.tensor(eidx, dtype=torch.int64)
    paper_author = torch.tensor(paper_author, dtype=torch.int64)

    v_weight = torch.ones(len(nodes), 1, dtype=torch.float32)
    e_weight = torch.ones(len(hyperedges), 1, dtype=torch.float32)

    args = argparse.Namespace()
    args.paper_author = paper_author
    args.n_hidden = 128
    args.predict_edge = False
    args.n_cls = 2
    args.edge_linear = False
    args.input_dim = len(nodes)
    args.n_layers = 2
    args.dropout_p = 0.3
    args.v_reg_weight = torch.Tensor([1.0])
    args.e_reg_weight = torch.Tensor([1.0])
    args.v_reg_sum = torch.Tensor([1.0])
    args.e_reg_sum = torch.Tensor([1.0])
    args.nv = nv
    args.ne = ne

    model = hh.Hypergraph(vidx, eidx, nv, ne, v_weight, e_weight, args)

    v_init = torch.randn(args.nv, args.input_dim)
    e_init = torch.zeros(args.ne, args.n_hidden)

    v_embeddings, e_embeddings, predictions = model(v_init, e_init)

    node_embeddings = v_embeddings.detach().numpy()
    print(f'done with {i}')
    embedding_nodes = pd.DataFrame(node_embeddings)
    embedding_nodes['Gene'] = h.get_nodes()
    embedding_nodes['Pathway'] = [pathway for _ in range(len(embedding_nodes))]
    dfs.append(embedding_nodes)

    i += 1

df = pd.concat(dfs, ignore_index=True)
df_names = [idx_to_name[idx] for idx in df['Gene']]
df['Gene name'] = df_names

cols_to_move = ['Gene', 'Gene name', 'Pathway']
new_order = cols_to_move + [col for col in df.columns if col not in cols_to_move]
df = df[new_order]
df = df.sort_values(by=['Gene', 'Pathway']).reset_index().iloc[:, 1:]

df.to_csv('results/embeddings/HNHN_128d.csv', index=False)