# Gene Dependency Analysis

[![Python Version](https://img.shields.io/badge/python-3.8%2B-blue)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-MIT-green)](LICENSE)

This repository contains the code and data for the MSc dissertation titled "Higher-Order Interactions in Gene Dependency," submitted for the University of Oxford and conducted in collaboration with Novo Nordisk. The project focuses on analyzing gene dependencies using advanced computational methods to uncover higher-order interactions.

## Table of Contents

- [Overview](#overview)
- [Repository Structure](#repository-structure)
- [Datasets](#datasets)
- [Installation](#installation)
- [Usage](#usage)
- [Results](#results)
- [Contributing](#contributing)
- [License](#license)

## Overview

Understanding gene dependencies is crucial for identifying potential therapeutic targets and understanding disease mechanisms. This project employs graph- and hypergraph-based and machine learning (ML) approaches to model and analyse complex gene interactions, aiming to identify higher-order dependencies that are not apparent through pairwise analyses.

## Repository Structure

The repository is organized as follows:

- `datasets/`: Contains raw and processed data files used in the analysis.
- `general`: `clustering.py` includes functions to apply spectral clustering to a graph with its optimal number of clusters iteratively until the order of every cluster is below a certain number. `utils.py` contains generic utilies used throughout the project, and `network_construction.py` has functions that perform graph- or hypergraph-related functionalities and are used throughout the whole project.
- `data_processing/` contains three scripts that prepare the CRISPR Gene Dependency data, the CORUM complexes data, and the Reactome pathway-to-gene mapping.
- `graph/`: Scripts for constructing and analysing gene interaction graphs. `graph_analysis.py` uses graph-related utilies from `graph_utils.py` to compute the edge density ratio of different CORUM complexes for a certain correlation threshold. This allows for an optimal threshold selection. `fitness_profile_plots.py` produces plots to show that correlation can be a measure of functional similarity.
- `hypergraph/`: Contains code for building and evaluating hypergraphs. `hyperedge_builder.py` clusters the input graph (with the selected threshold from the `graph/` directory) according to biological pathway and using spectral clustering and then infers the hyeredges for each cluster using the functions from `hyperedge_extraction.py`.
- `ml_model/`: Machine learning models for predicting gene dependencies. `feature_extraction.py` computes and saves the features used for ML, consisting of topological metrics and node embeddings. The topological metrics' computation for the hypergraphs has a custom implementation. `essentiality.py` loads the ML features and trains various ML models over a grid of hyperparameters, and then saves the performances of each model, including the f1-score, precision, recall, support, and accuracy.
- `pics/1-graph/`: Visualizations and figures generated during the project.

## Datasets

The `datasets` directory includes:

- **Raw Data**: Original datasets obtained from various sources.
- **Processed Data**: Data that has been cleaned and transformed for analysis.

*Note: Due to licensing restrictions, some datasets may not be included in this repository. Please refer to the respective data providers for access.*

## Installation

To set up the environment for this project:

1. **Clone the repository**:

   ```bash
   git clone https://github.com/IsmaLeal/gene_dependency.git
   cd gene_dependency
   ```

2. **Create a virtual conda environment** (recommended for using the graph-tool package.):
   
   ```bash
   conda create --name gene_dependency python=3.8
   conda activate gene_dependency
   ```

3. **Install the required packages**
   
   ```bash
   pip install -r requirements.txt
   ```

4. **Install graph-tool**

   Graph-tool is a non-PyPl package, so it must be installed manually via conda. For Debian/Ubuntu:
    ```bash
    sudo apt-get install python3-graph-tool
    ```
   See https://graph-tool.skewed.de/ for further installation instructions.

## Usage

1. **Data preparation**: Place the necessary datasets into the `datasets/` directory
2. **Data processing**: Run the files in `data_processing.py` to prepare the data for posterior analysis.
3. **Graph construction**: Run `graph_analysis.py` to compute the internal-to-external edge density ratios of CORUM protein complexes within the constructed graph. The script performs significance testing by comparing each complex's edge density ratio to a null distribution obtained via a configuration model, returning adjusted p-values. Complexes with statistically significant p-values validate the biological relevance of the constructed graph.
4. **Hypergraph construction**: Run `hypergraph_builder.py` to cluster the graph and save a file with the inferred hyperedges for each cluster
5. **ML models**: First run `feature_extraction.py` to save the appropriate features in the `ml_model/features/` directory. Then, `essentiality.py` runs various ML models and saves their performances in `ml_model/ml_results/`.

## License
This project is licensed under the MIT License. See the LICENSE file for details.
