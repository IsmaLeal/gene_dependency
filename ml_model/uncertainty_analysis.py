import os
import logging
import numpy as np
import pandas as pd
import graph_tool.all as gt
from sklearn.svm import SVC
from typing import Tuple, List
from general.utils import safe_read_csv
from sklearn.metrics import accuracy_score
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.utils.class_weight import compute_class_weight

# Paths
results_path = "./ml_results/"  # ML model performances
features_path = "./features/"   # ML features (topological metrics and embeddings)
visuals_path = "./visualisations/"
labels_path = "../datasets/CRISPRInferredCommonEssentials.csv"
corum_path = "../datasets/humanComplexes.txt"
os.makedirs(visuals_path, exist_ok=True)

# Loggging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


def load_best_model():
    """
    Reads model evaluation results and selects the best model based on accuracy.

    Returns
    -------
    best_dataset : str
        Name of the best features (e.g., "Graph Metrics", "Hypergraph Embeddings").
    best_model : str
        Name of the best ML model type (e.g., "Random Forest").
    """
    logging.info("Loading ML model performances...")

    model_performances = {}

    for file in os.listdir(results_path):
        if file.endswith(".csv"):
            df = safe_read_csv(os.path.join(results_path, file))
            best_acc = df[df["Label"] == "accuracy"]["accuracy"].max()
            best_model = df[df["accuracy"] == best_acc]["Model"].values[0]
            dataset_name = file.replace("_ml_performance.csv", "").replace("_", " ").title()
            model_performances[dataset_name] = (best_model, best_acc)

    # Select the best-performing model
    best_dataset = max(model_performances, key=lambda k: model_performances[k][1])
    best_model, _ = model_performances[best_dataset]

    logging.info(f"Best model: {best_model} on {best_dataset}")

    return best_dataset, best_model


def load_data(dataset_name: str) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    """
    Loads the features corresponding to the input `dataset_name`.

    This function loads the feature file corresponding to `dataset_name` and returns a tuple
    containing the features, the labels, and a DataFrame with both merged.

    Parameters
    ----------
    dataset_name : str
        String with the name of the features to be extracted. The feature files saved in
        `feature_extraction.py` allow for the following values of `dataset_name`:
        - "Graph Metrics".
        - "Hypergraph Metrics".
        - "Graph Embeddings"
        - "Hypergraph Embeddings"

    Returns
    -------
    X : pd.DataFrame
        Feature matrix corresponding to `dataset_name`.
    y : pd.Series
        Essentiality labels for each gene-pathway pair, according to CRISPR's Common Inferred
        Essentials in "../datasets/CRISPRInferredCommonEssentials.csv"
    df : pd.DataFrame
        DataFrame containing the features and labels for each gene-pathway pair.

    Raises
    ------
    FileNotFoundError
        If the feature file or the CRISPR Common Essentials are not found.
    """
    dataset_file = os.path.join(features_path, f"{dataset_name.replace(' ', '_').lower()}.csv")

    # Load labels vzdf
    df = safe_read_csv(dataset_file)
    labels = safe_read_csv(labels_path)
    if len(df) == 0 | len(labels) == 0:
        raise FileNotFoundError(f"Feature file or Essentiality dataset missing."
                                f"\nEnsure the files {os.path.join(features_path, dataset_name.replace(' ', '_').lower())}.csv"
                                f"\n\t and {labels_path} both exist.")

    df["Essential"] = df["Gene name"].isin(labels["Essentials"]).astype(int)

    # Select features and labels
    X = df.drop(columns=["Gene", "Gene name", "Pathway", "Essential"])
    y = df["Essential"]

    logging.info(f"Loaded dataset: {dataset_name} ({X.shape[0]} samples, {X.shape[1]} features).")

    return X, y, df


def train_and_predict(
        X: pd.DataFrame, y: pd.Series, best_model: str
) -> Tuple[np.ndarray, np.ndarray, pd.Series, pd.DataFrame]:
    """
    Trains the best model and returns predictions and class probabilities.

    This function splits the features and labels into a training and a testing set using the
    sklearn library, and then trains a ML model according to `best_model`.

    Parameters
    ----------
    X : pd.DataFrame
        DataFrame containing only the ML features.
    y : pd.Series
        Labels of the features in `X`.
    best_model : str
        Name of the best model. Possible options:
        - "Support Vector Classifier".
        - "Neural Network".
        - "Logistic Regression".
        - "Random Forest".
        - "KNN".

    Returns
    -------
    y_pred : np.ndarray
        Predicted class labels for X_test.
    probs : np.ndarray
        Class probabilities for X_test (if applicable, else None).
    y_test : pd.Series
        True labels for X_test.
    X_test : pd.DataFrame
        Feature set used for testing.

    Raises
    ------
    ValueError
        If `best_model` is not a recognised model name.
    TypeError
        If `params` contains invalid parameters for the chosen model.
    """
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=143)

    # Class weights
    class_weights = compute_class_weight("balanced", classes=np.unique(y_train), y=y_train)
    class_dict = dict(zip(np.unique(y_train), class_weights))

    # Model selection
    # Hyperparameter grids
    param_grid_svc = {
        'class_weight': [class_dict],
        'gamma': ['scale'],
        'kernel': ['poly', 'rbf'],
        'C': [1.0, 10.0]
    }
    param_grid_lr = {
        'class_weight': [class_dict],
        'solver': ['lbfgs', 'liblinear', 'newton-cholesky'],
        'C': [0.1, 1.0, 10.0],
        'max_iter': [5000]
    }
    param_grid_rf = {
        'n_estimators': [100, 200, 300, 400, 500],
        'class_weight': [class_dict]
    }
    param_grid_nn = {
        'hidden_layer_sizes': [(100, 100), (100, 75, 75)],
        'max_iter': [500]
    }
    param_grid_knn = {
        'n_neighbors': [1, 3, 5, 7, 10, 15]
    }
    model_dict = {
        "support vector classifier": (SVC, param_grid_svc),
        "neural network": (MLPClassifier, param_grid_nn),
        "logistic regression": (LogisticRegression, param_grid_lr),
        "random forest": (RandomForestClassifier, param_grid_rf),
        "knn": (KNeighborsClassifier, param_grid_knn)
    }

    if best_model.lower() not in model_dict:
        raise ValueError(f"Invalid `best_model` name: {best_model}."
                         f"\nChoose from {list(model_dict.keys())}.")

    ModelClass, param_grid = model_dict[best_model.lower()]

    # Try initialising the model with the input parameters
    try:
        grid_search = GridSearchCV(ModelClass, param_grid, cv=5, scoring="accuracy")
    except TypeError as e:
        raise TypeError(f"Invalid hyperparameters for {best_model}: {e}.")

    # Train model
    logging.info(f"Training model ({best_model}).")
    grid_search.fit(X_train, y_train)
    model = grid_search.best_estimator_

    # Obtain predictions
    y_pred = model.predict(X_test)
    probs = model.predict_proba(X_test) if hasattr(model, "predict_proba") else None

    logging.info(f"Model accuracy: {accuracy_score(y_test, y_pred):.2f}.")

    return y_pred, probs, y_test, X_test


def gene_prediction_confidence(
        df: pd.DataFrame, y_pred: np.ndarray, probs: np.ndarray, y_test: np.ndarray, mode: str = "uncertain"
) -> Tuple[pd.DataFrame, List]:
    """
    Identifies genes with the most uncertain or certain predictions across pathways.

    This function finds the confidence of every essentiality prediction, one for each gene-pathway pair,
    and counts the number of high-/low-confidence predictions that each gene has. It then selects and
    returns the top 20 genes with more high-/low-cofidence predictions.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing gene index, gene name, and pathway name for each gene-pathway pair
        in the test set.
    y_pred : np.ndarray
        Array with the essentiality predictions for the test set.
    probs : np.ndarray
        Class probabilities for X_test (if applicable, else None).
    y_test : np.ndarray
        Array with the essentiality labels from the CRISPR Inferred Common Essentials dataset.
    mode : str, optional
        Determines whether to compute the most uncertain or certain genes. Options:
        - "uncertain": Genes with more number of lower-confidence predictions.
        - "certain": Genes with more number of high-confidence predictions.

    Returns
    -------
    confidence_data : pd.DataFrame
        DataFrame containing each gene-pathway pair with its essentiality predictions and labels, together
        with the probability associated to each prediction (if applicable).
    top_20_genes : list
        List containing the top 20 genes with the largest number of either uncertain or certain predictions.

    Raises
    ------
    ValueError
        If `mode` is not "uncertain" or "certain".
    """
    if mode not in ["uncertain", "certain"]:
        raise ValueError(f"Invalid mode '{mode}'. Choose 'uncertain' or 'certain'.")
    if mode == "uncertain":
        p_min, p_max = 0.5, 0.55    # Low confidence range
    else:
        p_min, p_max = 0.85, 1.00   # High confidence range

    nodes = df["Gene"]
    names = df["Gene name"]
    pathways = df["Pathway"]

    # Identify predictions of interest
    confidence_flags = np.zeros(len(y_test))
    for idx, p in enumerate(probs[:, 0]):   # Class 0 probabilities
        if p_min <= p <= p_max:
            confidence_flags[idx] = 1
    for idx, p in enumerate(probs[:, 1]):   # Class 1 probabilities
        if p_min <= p <= p_max:
            confidence_flags[idx] = 1

    # Create a DataFrame with confidence data
    confidence_data = pd.DataFrame({
        "Gene": nodes,
        "Gene name": names,
        "Pathway": pathways,
        "Prediction": y_pred,
        "Label": y_test,
        f"{mode.capitalize()} confidence": confidence_flags
    })

    # Sum low-confidence counts for every gene
    gene_uncertainty = confidence_data.groupby("Gene name")["Low confidence"].sum().sort(ascending=False)
    top_20_genes = gene_uncertainty.head(20).index.tolist()

    logging.info(f"Top 20 genes with most uncertain predictions: {top_20_genes}")

    return confidence_data, top_20_genes


def visualise_uncertainty_graph(uncertain_data: pd.DataFrame, top_20_genes: List) -> None:
    """
    Visualizes a bipartite graph of top uncertain genes and their pathways.

    The output graph shows genes as blue nodes and pathways as orange nodes. Edges only connect
    genes and pathways if a gene belongs to the pathway. Edges are coloured green if the essentiality
    prediction agrees with the CRISPR Inferred Common Essentials dataset, and red otherwise.

    Parameters
    ----------
    uncertain_data : pd.DataFrame
        DataFrame containing each gene-pathway pair with its essentiality predictions and labels, together
        with the probability associated to each prediction (if applicable).
    top_20_genes : list
        List containing the top 20 genes with the largest number of uncertain predictions.
    """
    logging.info("Building uncertainty graph.")

    subgraph = gt.Graph(directed=False)

    sub_v_prop_name = subgraph.new_vertex_property("string")
    sub_v_prop_type = subgraph.new_vertex_property("string")
    sub_e_prop_correct = subgraph.new_edge_property("bool")

    sub_vertex_dict = {}
    pathways_20 = []

    for _, row in uncertain_data.iterrows():
        gene, pathway, pred, label = row["Gene"], row["Pathway"], row["Prediction"], row["Label"]

        if row["Gene name"] in top_20_genes:
            pathways_20.append(pathway)

            if gene not in sub_vertex_dict:
                v_gene = subgraph.add_vertex()
                sub_vertex_dict[gene] = v_gene
                sub_v_prop_name[v_gene] = row["Gene name"]
                sub_v_prop_type[v_gene] = "gene"

            if pathway not in sub_vertex_dict:
                v_pathway = subgraph.add_vertex()
                sub_vertex_dict[pathway] = v_pathway
                sub_v_prop_name[v_pathway] = pathway
                sub_v_prop_type[v_pathway] = "pathway"

            edge = subgraph.add_edge(sub_vertex_dict[gene], sub_vertex_dict[pathway])
            sub_e_prop_correct[edge] = (pred == label)

    subgraph.vertex_properties["name"] = sub_v_prop_name
    subgraph.vertex_properties["type"] = sub_v_prop_type
    subgraph.edge_properties["correct"] = sub_e_prop_correct

    # Visualization
    pos = gt.sfdp_layout(subgraph, C=2)
    output_file = os.path.join(visuals_path, "uncertainty_graph.png")

    gt.graph_draw(subgraph, pos,
                  vertex_fill_color="black",
                  vertex_text=sub_v_prop_name,
                  vertex_text_position=1,
                  vertex_font_size=6,
                  vertex_text_color="black",
                  vertex_text_background="white",
                  vertex_size=6,
                  edge_color="gray",
                  output_size=(600, 600),
                  output=output_file)

    logging.info(f"Uncertainty graph saved: {output_file}")


def match_protein_complexes(confident_data: pd.DataFrame):
    """
    Checks if hyperedges from pathways associated with high-confidence gene-pathway predictions
    are part of CORUM complexes.

    HOWWW DO WE EXTRACT CERTAIN DATAAA

    Parameters
    ----------
    confident_data : pd.DataFrame
        DataFrame with gene-pathway pairs that had high-confidence predictions.

    Returns
    -------
    matching_complexes : pd.DataFrame
        DataFrame with CORUM complexes.
    """
    corum_df = safe_read_csv(corum_path, delimiter="\t")
    corum_df = corum_df[["ComplexID", "subunits(Gene name)", "Cell line"]]

    corum_complexes = [
        (frozenset(genes.split(";")), cid, cell_line)
        for cid, genes, cell_line in zip(
            corum_df["ComplexID"], corum_df["subunits(Gene name)"], corum_df["Cell line"]
        )
    ]

    # Load hyperedges
    hyperedges = safe_read_csv("../datasets/hyperedges_final.csv")
    hyperedges["Hyperedges (gene names)"] = hyperedges["Hyperedges (gene names)"].apply(eval)

    total = 0
    matched = 0
    matched_complexes = []

    # Iterate over confident gene-pathway pairs
    logging.info(f"Checking gene-pathway pairs...")
    for _, row in confident_data.iterrows():
        gene = row["Gene name"]
        pathway = row["Pathway"]

        # Get hyperedges in this pathway
        pathway_edges = hyperedges[hyperedges["Pathway"] == pathway]["Hyperedges (gene names)"]
        if pathway_edges.empty:
            continue

        for edge in pathway_edges.values[0]:
            if gene not in edge:    # Skip if this gene isn't in this hyperedge
                continue

            total += 1
            for complex_genes, complex_id, cell_line in corum_complexes:
                if complex_genes.issubset(edge):    # Check if CORUM complex is within the hyperedge
                    if gene not in complex_genes:   # Ensure gene is in the complex
                        continue
                    matched += 1
                    matched_complexes.append({
                        "Gene": gene,
                        "Pathway": pathway,
                        "ComplexID": complex_id,
                        "Complex Genes": ";".join(complex_genes),
                        "Cell line": cell_line
                    })
                    break

    matched_df = pd.DataFrame(matched_complexes)

    logging.info(f"From {total} hyperedges, found {matched} matching protein complexes.")

    if not matched_df.empty:
        import ace_tools as tools
        tools.display_dataframe_to_user(name="Matched Protein Complexes", dataframe=matched_df)

    return matched_df



if __name__ == "__main__":
    dataset_name, best_model = load_best_model()
    X, y, df = load_data(dataset_name)
    y_pred, probs, y_test, X_test = train_and_predict(X, y, best_model)
    df_test = df.iloc[X_test.index].reset_index(drop=True)
    uncertain_data, top_20_genes = gene_prediction_confidence(df_test, y_pred, probs, y_test, mode="uncertain")
    visualise_uncertainty_graph(uncertain_data, top_20_genes)
    confident_data, _ = gene_prediction_confidence(df_test, y_pred, probs, y_test, mode="certain")
    matched_df = match_protein_complexes(confident_data)
