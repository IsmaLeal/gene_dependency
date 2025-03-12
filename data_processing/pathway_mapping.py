import os
import warnings
import numpy as np
import pandas as pd

def map_pathway_to_nodes() -> pd.DataFrame:
    """
    Maps each pathway to the corresponding nodes.

    This function returns a dataframe mapping each pathway to the nodes (names & indices) it contains.
    Includes two clusterings apart from the pathways extracted from UniProt:
    - "Unmapped": includes all genes with either NaN pathways or belonging to a single-gene pathway.
    - "Unknown": includes all genes from CRISPR not belonging to the UniProt dataset.
    """
    data_path = "../datasets/"

    genes = pd.read_csv(os.path.join(data_path, "names.txt"), header=None)[0]
    gene_to_index = {name: idx for idx, name in enumerate(genes)}

    # Load the UniProt files
    dfs = []
    for i in range(4):
        warnings.simplefilter("ignore", category=UserWarning)   # Suppress OpenPyXL warnings
        df = pd.read_excel(os.path.join(data_path, f"uniprot/uniprot{i + 1}.xlsx"))
        dfs.append(df)
    df = pd.concat([i for i in dfs], ignore_index=True)
    df = df[df["Reviewed"] == "reviewed"].reset_index().iloc[:, 1:]
    df = df[["From", "Reactome"]]  # Select columns with gene name and pathway name

    # Remove NaN values and add them to the unmapped group
    unmapped = df[df["Reactome"].isna()]
    unmapped_list = list(np.unique(unmapped["From"].values))
    df = df.dropna(subset=["Reactome"]).reset_index().iloc[:, 1:]

    # Create an extended dataframe with a 1 to 1 mapping from every gene to all the pathways it belongs to
    name_list, pathway_list = [], []
    for idx, row in df.iterrows():
        pathways = str(df.iloc[idx, 1])
        pathways = [pathway for pathway in pathways.split(";") if pathway != ""]
        for pathway in pathways:
            name_list.append(df.iloc[idx, 0])
            pathway_list.append(pathway)
    extended = pd.DataFrame({"Genes": name_list, "Pathway": pathway_list})

    # Group by pathway
    pathway_to_genes = extended.groupby("Pathway")["Genes"].agg(list).reset_index()
    # We'll take single-gene pathways to be in the same classification as those genes without a pathway
    unmapped = pathway_to_genes[[len(genes) < 2 for genes in pathway_to_genes["Genes"].values]]
    unmapped_list.extend([gene[0] for gene in np.unique(unmapped["Genes"].values)])
    # Classify the rest of the CRISPR genes as unknown
    unknown_list = [gene for gene in genes if (not gene in extended["Genes"].values)
                    & (not gene in unmapped_list)]
    pathway_to_genes = pathway_to_genes[
                           [len(genes) > 1 for genes in pathway_to_genes["Genes"].values]].reset_index().iloc[:, 1:]
    # Add "unmapped" and "unknown" to the DF
    pathway_to_genes.loc[len(pathway_to_genes)] = pd.Series({"Pathway": "no_pathway", "Genes": unmapped_list})
    pathway_to_genes.loc[len(pathway_to_genes)] = pd.Series({"Pathway": "unknown_pathway", "Genes": unknown_list})

    # Add node indices
    nodes = []
    for idx, row in pathway_to_genes.iterrows():
        pathway_nodes = [gene_to_index[gene] for gene in row["Genes"] if gene in gene_to_index]
        nodes.append(pathway_nodes)
    pathway_to_genes["Nodes"] = nodes

    # Remove duplicate pathways consisting of a set of nodes already present in the DF
    df = pathway_to_genes.drop_duplicates(subset="Nodes", keep="first").reset_index().iloc[:, 1:]
    df["num_genes"] = df["Nodes"].apply(len)

    df.to_csv(os.path.join(data_path, "pathway_mapping.csv"), index=False)

    return df


# Run the function
if __name__ == "__main__":
    map_pathway_to_nodes()
