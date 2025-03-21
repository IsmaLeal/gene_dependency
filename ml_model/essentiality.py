import os
import logging
import numpy as np
import pandas as pd
from sklearn.svm import SVC
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.utils.class_weight import compute_class_weight
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, ConfusionMatrixDisplay

# Configuration
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
features_path = "./features/"
labels_path = "../datasets/CRISPRInferredCommonEssentials.csv"
results_path = "./ml_results/"
os.makedirs(results_path, exist_ok=True)

# Feature filenames
feature_files = {
    "Graph Metrics": "graph_metrics.csv",
    "Graph Embeddings": "graph_embeddings.csv",
    "Hypergraph Metrics": "hypergraph_metrics.csv",
    "Hypergraph Embeddings": "hypergraph_embeddings.csv"
}


def plot_confusion_matrix(y_true, y_pred, model_name):
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot(cmap=plt.cm.Blues)
    plt.title(f"Confusion Matrix for {model_name}")
    plt.show()


def check_class_balance(y):
    unique, counts = np.unique(y, return_counts=True)
    balance = dict(zip(unique, counts))
    return balance


def get_class_weights(y):
    classes = np.unique(y)
    class_weights = compute_class_weight("balanced", classes=classes, y=y)
    return dict(zip(classes, class_weights))


def evaluate_models(X, y, models, features_name: str):
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    results = {}

    # Perform grid search
    for name, (model, param_grid) in models.items():
        logging.info(f"Performing 5-fold CV for model {name}")
        grid_search = GridSearchCV(model, param_grid, cv=5, scoring="accuracy")
        grid_search.fit(X_train, y_train)

        logging.info(f"Best parameters found for {name} ({features_name}): {grid_search.best_params_}")
        best_model = grid_search.best_estimator_
        y_pred = best_model.predict(X_test)

        logging.info(f"Accuracy for {name} ({features_name}): {accuracy_score(y_test, y_pred):.2f}")
        class_report = classification_report(y_test, y_pred, output_dict=True)
        plot_confusion_matrix(y_test, y_pred, name)
        print("-" * 60)
        results[name] = class_report

    return results


# Load CRISPR essentiality labels
labels = pd.read_csv(labels_path)

# ------------------------------------------------------------------------
datasets = {}
for dataset_name, file_name in feature_files.items():
    df = pd.read_csv(os.path.join(features_path, file_name))
    df["Essential"] = df["Gene name"].isin(labels["Essentials"]).astype(int)
    datasets[dataset_name] = df

results = {}
for dataset_name, df in datasets.items():
    # Define the features and the target
    X = df.drop(columns=['Gene', 'Gene name', 'Pathway', 'Essential'])
    y = df['Essential']

    # Check class balance and compute class weights
    logging.info(f"\nProcessing: {dataset_name}.\nClass balance: {check_class_balance(y)}.")
    class_dict = get_class_weights(y)

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

    models = {
        "Support Vector Classifier": (SVC(random_state=143), param_grid_svc),
        "Neural Network": (MLPClassifier(random_state=143), param_grid_nn),
        "Logistic Regression": (LogisticRegression(random_state=143), param_grid_lr),
        "Random Forest": (RandomForestClassifier(random_state=143), param_grid_rf),
        "K-Nearest Neighbors": (KNeighborsClassifier(), param_grid_knn)
    }

    # Train models on current dataset
    results[dataset_name] = evaluate_models(X, y, models, dataset_name)

# Store results for each dataset
for dataset_name, model_results in results.items():
    records = []
    for model, metrics in model_results.items():
        for key, scores in metrics.items():
            if isinstance(scores, dict):
                record = {'Model': model, 'Label': key}
                record.update(scores)
                records.append(record)
            else:  # Accuracy
                record = {'Model': model, 'Label': key, 'f1-score': None, 'precision': None, 'recall': None,
                          'support': None, 'accuracy': scores}
                records.append(record)

    df = pd.DataFrame(records)
    output_file = os.path.join(results_path, f"{dataset_name.replace(' ', '_').lower()}_ml_performance.csv")
    df.to_csv(output_file, index=False)
