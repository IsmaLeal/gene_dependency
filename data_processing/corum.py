import pandas as pd
from typing import List

def filter_CORUM() -> pd.DataFrame:
    """
    Filters the CORUM dataset for relevant protein complexes based on pre-defined cell line names.

    Returns
    -------
    pd.DataFrame
        Filtered dataset of protein complexes.
    """
    # Load CORUM dataset
    file_path = "../datasets/humanComplexes.txt"
    try:
        df = pd.read_csv(file_path, delimiter="\t")
    except:
        print(f"File {file_path} does not exist. Please re-download from CORUM website")

    # Select rows containing these substrings in their 'Cell line' value
    substrings = ["T cell line ED40515",
                  "mucosal lymphocytes",
                  "CLL cells",
                  "monocytes",
                  "THP-1 cells",
                  "bone marrow-derived",
                  "monocytes, LPS-induced",
                  "THP1 cells",
                  "human blood serum",
                  "human blood plasma",
                  "plasma",
                  "CSF",
                  "human leukemic T cell JA3 cells",
                  "erythrocytes",
                  "peripheral blood mononuclear cells",
                  "African Americans",
                  "SKW 6.4",
                  "BJAB cells",
                  "Raji cells",
                  "HUT78",
                  "J16",
                  "H9",
                  "U-937",
                  "Jurkat T",
                  "NB4 cells",
                  "U937",
                  "early B-lineage",
                  "T-cell leukemia",
                  "lymphoblasts",
                  "whole blood and lymph",
                  "human neutrophil-differentiating HL-60 cells",
                  "human peripheral blood neutrophils",
                  "human neutrophils from fresh heparinized human peripheral blood",
                  "human peripheral blood",
                  "HCM",
                  "liver-hematopoietic",
                  "cerebral cortex",
                  "human brain",
                  "pancreatic islet",
                  "human hepatocyte carcinoma HepG2 cells",
                  "Neurophils",
                  "H295R adrenocortical",
                  "frontal cortex",
                  "myometrium",
                  "vascular smooth muscle cells",
                  "Dendritic cells",
                  "intestinal epithelial",
                  "Primary dermal fibroblasts",
                  "HK2 proximal",
                  "brain pericytes",
                  "HepG2",
                  "HEK 293 cells, liver",
                  "normal human pancreatic duct epithelial",
                  "pancreatic ductal adenocarcinoma",
                  "OKH cells",
                  "cultured podocytes",
                  "renal glomeruli",
                  "VSMCs",
                  "differentiated HL-60 cells",
                  "SH-SY5Y cells",
                  "frontal and entorhinal cortex",
                  "SHSY-5Y cells",
                  "hippocampal HT22 cells",
                  "primary neurons",
                  "neurons",
                  "renal cortex membranes",
                  "Kidney epithelial cells",
                  "skeletal muscle cells",
                  "Skeletal muscle fibers",
                  "differentiated 3T3-L1",
                  "brain cortex",
                  "cortical and hippocampal areas",
                  "human H4 neuroglioma",
                  "Thalamus",
                  "HISM",
                  "pancreas",
                  "RCC4",
                  "C2C12 myotube",
                  "XXVI muscle",
                  "SH-SY5Y neuroblastoma",
                  "HCC1143",
                  "Hep-2",
                  "PANC-1",
                  "HEK293T cells",
                  "HEK-293 cells",
                  "heart",
                  "epithelium",
                  "kidney",
                  "heart muscle",
                  "central nervous system",
                  "COS-7 cells",
                  "ciliary ganglion",
                  "striated muscle",
                  "PC12",
                  "293FR cells"]
    pattern = "|".join(substrings)

    # Select rows whose 'Cell line' value is exactly one of these
    exact = ["muscle", "293 cells", "brain", "HEK 293 cells"]

    # Create Boolean mask selecting all the rows described
    partial_mask = df["Cell line"].str.contains(pattern, case=False, na=False)
    exact_mask = df["Cell line"].isin(exact)
    total_mask = partial_mask | exact_mask

    # Obtain filtered dataframe, select relevant columns, sort by 'Cell line'
    complexes_full = df[total_mask]
    complexes = complexes_full[
        ["ComplexID", "ComplexName", "Cell line", "subunits(Gene name)", "GO description", "FunCat description"]]
    complexes = complexes.sort_values(by=["Cell line"])

    # Exclude those complexes including only one subunit/ gene
    mask_mono = [len(complexes["subunits(Gene name)"].values[i].split(";")) > 1 for i in range(len(complexes))]
    complexes = complexes.loc[mask_mono]
    return complexes


def load_CORUM() -> List[str]:
    """
    Loads the filtered CORUM dataset from '../datasets/filtered_complexes.csv'.

    Returns
    -------
    complexes : List[str]
        List of protein complexes (each complex is a list of gene names) or None
        if the file is missing.

    Examples
    --------
    >>> complexes = load_CORUM()
    >>> print(complexes[0])
    ["GeneA", "GeneB", "GeneC"]
    """
    file_path = "../datasets/filtered_complexes.csv"
    try:
        df = pd.read_csv(file_path)
        complexes_strings = df['subunits(Gene name)'].values
        complexes = [complex.split(";") for complex in complexes_strings]
    except:
        print(f"File {file_path} does not exist. Please run filter_CORUM().")
        complexes = None
    return complexes


if __name__ == "__main__":
    filter_CORUM()
