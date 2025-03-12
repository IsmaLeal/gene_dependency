import os
import ast
import logging
import pandas as pd
from general.utils import safe_read_csv
from hyperedge_extraction import extract_hyperedges
from general.clustering import iterative_spectral_clustering
from general.graph_construction import create_graph_networkx, create_subgraphs

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

if __name__ == "__main__":
    # Configuration
    data_path = "../datasets/"
    threshold = 0.2
    num_clusters = 3
    mcmc_sweeps = 1000

    # Load graph
    g = create_graph_networkx(threshold)

    genes = pd.read_csv(os.path.join(data_path, "names.txt"), header=None)[0]
    idx_to_gene = {idx: name for idx, name in enumerate(genes)}

    # Cluster the NetworkX Graph object
    pathway_mapping = safe_read_csv(os.path.join(data_path, "pathway_mapping.csv"))
    if not pathway_mapping:
        raise FileNotFoundError(
            f"The file `../datasets/pathway_mapping.csv` is missing."
            f"Run `map_pathway_to_nodes()` from `../data_processing/pathway_mapping.py` first."
        )
    pathway_subgraphs = create_subgraphs(g)

    # Initialise dictionary for storing hyperedges
    all_hyperedges = {}
    all_pathways = {}
    all_num_hyperedges = {}
    all_num_nodes = {}

    max_size = 1300

    for pathway, subgraph in pathway_subgraphs.items():
        logging.info(f"Processing pathway: {pathway} ({subgraph.number_of_nodes()} nodes, {subgraph.number_of_edges()} edges)")

        # Apply spectral clustering only to pathways large enough to be subdivided
        if subgraph.number_of_nodes() > max_size:
            clusters = iterative_spectral_clustering(subgraph, max_size=max_size)
        else:
            clusters = {pathway: subgraph}

        hyperedges = extract_hyperedges(clusters)
        all_hyperedges[pathway] = hyperedges
        all_pathways = None

    results = []

    for pathway, hyperedge_dict in all_hyperedges.items():
        for label, edges in hyperedge_dict.items():
            try:
                int_label = int(label)
                pathway_name = f"no_pathway_{int_label}"
            except ValueError:
                pathway_name = pathway

            node_indices = ast.literal_eval(pathway_mapping[pathway_mapping["Pathway"] == pathway]["Nodes"].values[0])

            # Add gene names for each hyperedge too
            hyperedge_names = [
                tuple(idx_to_gene[node] for node in edge) for edge in edges
            ]

            # Store data
            results.append([
                pathway_name,           # Pathway Name
                str(node_indices),      # Node Indices
                len(node_indices),      # Number of Nodes
                str(hyperedge_names),   # Hyperedges (gene names)
                str(edges),             # Hyperedges (node indices)
                len(edges)              # Number of Hyperedges
            ])

        # Convert to DataFrame
        df_final = pd.DataFrame(results, columns=[
            "Pathway", "Nodes", "Number of Nodes",
            "Hyperedges (genes)", "Hyperedges (indices)", "Number of Hyperedges"
        ])

        # Save to CSV
        output_file = os.path.join(data_path, "hyperedges_final.csv")
        df_final.to_csv(output_file, index=False)
