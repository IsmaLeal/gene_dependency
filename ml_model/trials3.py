import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.utils.class_weight import compute_class_weight
from sklearn.svm import SVC, LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

def plot_confusion_matrix(y_true, y_pred, model_name):
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot(cmap=plt.cm.Blues)
    plt.title(f'Confusion Matrix for {model_name}')
    plt.show()


def check_class_balance(y):
    unique, counts = np.unique(y, return_counts=True)
    balance = dict(zip(unique, counts))
    print("Class balance:", balance)


def get_class_weights(y):
    classes = np.unique(y)
    class_weights = compute_class_weight('balanced', classes=classes, y=y)
    return dict(zip(classes, class_weights))


def evaluate_models(X, y, models, graph):
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    genes_train = X_train['Gene'].values
    pathways_train = X_train['Pathway'].values
    genes_test = X_test['Gene'].values
    pathways_test = X_test['Pathway'].values

    X_train = X_train.drop(columns=['Gene', 'Pathway'])
    X_test = X_test.drop(columns=['Gene', 'Pathway'])

    # rf_model = RandomForestClassifier(n_estimators=500, random_state=42)
    # rf_model.fit(X_train, y_train)
    #
    # y_pred = rf_model.predict(X_test)

    results = {}

    # Perform grid search
    for name, (model, param_grid) in models.items():
        print(f'Trying {name}')
        grid_search = GridSearchCV(model, param_grid, cv=5, scoring='accuracy')  # 5-fold cross-validation
        print('now fitting...')
        grid_search.fit(X_train, y_train)

        print(f"Best parameters for {name} ({graph}): {grid_search.best_params_}")
        best_model = grid_search.best_estimator_
        y_pred = best_model.predict(X_test)

        print(f"Accuracy for {name} ({graph}): {accuracy_score(y_test, y_pred):.2f}")
        class_report = classification_report(y_test, y_pred, output_dict=True)
        plot_confusion_matrix(y_test, y_pred, name)
        print("-" * 60)
        results[name] = class_report

    return results, y_pred, y_test, genes_test, pathways_test


hg_metrics = pd.read_csv('results/embeddings/g_embeddings_128d.csv')

# Load CRISPR essentiality labels
labels = pd.read_csv('../datasets/CRISPRInferredCommonEssentials.csv')

# Merge both
# hypergraph...
hg_metrics['Essential'] = hg_metrics['Gene name'].isin(labels['Essentials']).astype(int)

# Define the features and the target

X_hg = hg_metrics.drop(columns=['Gene name', 'Essential'])
y_hg = hg_metrics['Essential']

check_class_balance(y_hg)

# Class imbalance
classes = np.unique(y_hg)
class_weights = compute_class_weight('balanced', classes=classes, y=y_hg)
class_dict = dict(zip(classes, class_weights))

# Hyperparameter grids
# param_grid_svc = {
#     'class_weight': [class_dict],
#     'gamma': ['scale'],
#     'kernel': ['linear', 'rbf', 'poly'],
#     'C': [0.1, 1, 10]
# }
param_grid_svc = {
    'class_weight': [class_dict],
    'gamma': ['scale'],
    'kernel': ['poly', 'rbf'],
    'C': [10]
}

param_grid_lr = {
    'class_weight': [class_dict],
    'solver': ['lbfgs', 'liblinear'],
    'C': [0.1, 1.0, 10.0],   # Inverse of regularisation strength
    'max_iter': [5000]
}

param_grid_rf = {
    'n_estimators': [100, 200, 300, 400, 500],
    'class_weight': [class_dict]
}

# param_grid_nn = {
#     'hidden_layer_sizes': [(500, 400, 300, 200, 100)], #, (500, 250, 100)],
#     'max_iter': [600]
# }

param_grid_nn = {
    'hidden_layer_sizes': [(100, 75, 50), (100, 50)],
    'max_iter': [500]
}

param_grid_knn = {
    'n_neighbors': [1, 3, 5, 7, 10, 15]
}

# Models
# models = {
#     "Neural Network": (MLPClassifier(random_state=42), param_grid_nn)
# }
models = {
    "Support Vector Classifier": (SVC(random_state=42), param_grid_svc),
    "Neural Network": (MLPClassifier(random_state=42), param_grid_nn),
    "Logistic Regression": (LogisticRegression(random_state=42), param_grid_lr),
    "Random Forest": (RandomForestClassifier(random_state=42), param_grid_rf),
    "K-Nearest Neighbors": (KNeighborsClassifier(), param_grid_knn)
}

results_hg, y_pred, y, genes, pathways = evaluate_models(X_hg, y_hg, models, 'Graph')

# Store hypergraph results
records_hg = []
for model, metrics in results_hg.items():
    for key, scores in metrics.items():
        if isinstance(scores, dict):
            record = {'Model': model, 'Label': key}
            record.update(scores)
            records_hg.append(record)
        else:  # Accuracy
            record = {'Model': model, 'Label': key, 'f1-score': None, 'precision': None, 'recall': None,
                      'support': None, 'accuracy': scores}
            records_hg.append(record)

df_hg = pd.DataFrame(records_hg)
df_hg.to_csv('results/embeddings/g_128d_ml_performance.csv', index=False)