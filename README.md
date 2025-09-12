# Deep-pert

This is a framework implementation of Deep-pert, as described in our paper:

![demo](https://github.com/sirpan/Deep-pert/blob/main/Fig1.png)

## Contents

- [Overview](#overview)
- [Repo Contents](#repo-contents)
- [System Requirements](#system-requirements)
- [User guide](#User_guide)
# Overview

Effective treatment of complex diseases like cancer and MASH (Metabolic Dysfunction-Associated Steatotic Hepatitis) remains a challenge due to high attrition rates and disease heterogeneity. Although drug perturbation profiles offer rich functional data, current AI models are often unstable and lack the mechanistic interpretability needed for clinical translation. Here, we present Deep-pert, a deep ensemble learning framework that generates robust and mechanistically interpretable representations from high-dimensional pharmacogenomic data. Deep-pert significantly outperforms current methods in predicting drug sensitivity, synergy, and toxicity. Its interpretable features reveal disease-specific mechanisms and successfully predict validated combination therapies. We validate Deep-pert’s predictions in both cancer and MASH, identifying novel drug combinations with experimental support. Its versatility is further demonstrated through successful applications to drug-induced metabolomics, confirming its potential to guide precision medicine. Deep-pert provides a scalable platform for drug repurposing, biomarker discovery, and disease-specific pathway exploration, advancing therapeutic development in complex disease contexts.
# Repo Contents
- [Train](./Train): source code of Deep-pert for training in the paper.
- [Interpretability and ensembles](./Interpretability%20and%20ensembles): source code of Deep-pert for interpretability and ensembles in the paper.
- [Compute Target combination](./Compute%20Target%20combination): source code of Deep-pert for computing target combination in the paper.
- [down task](./down%20task): source code of Deep-pert for down tasks in the paper.
- [Results](./Results): data appled by deep-pert for all code.


# System Requirements

## code dependencies and operating systems

### code dependencies

Users should install the following packages first, which will install in about 30 minutes on a machine with the recommended specs. For detailed dependencies, see requirements.txt
```
python == 3.6.8
cuda ==10.1/11.7
```
### Operating systems
The package development version is tested on *Linux and Windows 10* operating systems. The developmental version of the package has been tested on the following systems:

Linux: Ubuntu 18.04  
Windows: 10

The pip package should be compatible with Windows, and Linux operating systems.

Before setting up the FINDER users should have `gcc` version 7.4.0 or higher.

## Hardware Requirements
The model requires a standard computer with enough RAM and GPU to support the operations defined by a user. For minimal performance, this will be a computer with about 4 GB of RAM and 16GB of GPU. For optimal performance, we recommend a computer with the following specs:

RAM: 16+ GB  
CPU: 4+ cores, 3.3+ GHz/core
GPU: 16+ GB

The runtimes below are generated using a computer with the recommended specs (16 GB RAM, 4 cores@3.3 GHz) and internet of speed 25 Mbps.

# User_guide
Before using the ComplexDnet software, please read the Directions for use.pdf in the root directory carefully

## Core functional modules

### 1. Deep learning model training (`train_all_models.py`)

**What it does**: Unified entry point for model training, supports end-to-end training of multiple deep learning models and baseline methods.

**Supported Models**.
- **Deep Learning Models**: AE (Auto-Encoder), DAE (Denoising Auto-Encoder), VAE (Variational Auto-Encoder).
- **Baseline methods**: PCA, ICA (Independent Component Analysis), RP (Random Projection)


**Usage**:
```bash
python train_all_models.py \
  --models VAE,AE,DAE \
  --cancer-types A549,HEPG2 \
  --latent-dims 5,10,25,50 \
  --runs 5 \
  --input-root /path/to/input \
  --output-root /path/to/output
```


**Key features**.
- Automatic PCA preprocessing
- Multiple cancer types support
- Multiple latent dimension configurations
- Multi-run support
- Automatic result counting and saving

### 2. Deep Learning Model Implementation (`unified_deep_models.py`)

**Role**: Provides unified implementation of three deep learning models: AE, DAE, and VAE.

**Model Architecture**.
- **AE**: standard auto-encoder, including encoder and decoder.
- **DAE**: Denoising self-encoder, injecting Gaussian noise during training.
- **VAE**: Variational self-encoder, supports latent space sampling.

**Core Functions**.
- Automatic network architecture construction
- Cosine annealing learning rate scheduling
- Beta-VAE implementation of VAE (WarmUp callback)
- Automatic model saving and evaluation

### 3. Interpretability analysis and integration (`run_interpretability_and_ensembles.py`)

**Role**: computes integration gradients, performs GMeans clustering, generates integration labels and weights.

**Main Functions**.
- **Integrated Gradient Calculation**: analyses model sensitivity to input features
- **GMeans clustering**: automatically determine the optimal number of clusters
- **Integrated Weight Generation**: merge gene importance weights from multiple models
- **Training Embedding Generation**: create embedding representations for downstream analysis

**Usage**:
```bash
python run_interpretability_and_ensembles.py \
  --models VAE,AE,DAE \
  --cancer-types A549 \
  --latent-dims 5,10,25,50 \
  --runs 3 \
  --input-root /path/to/input \
  --output-root /path/to/output \
  --pca-method PCA
```

**Output file**.
- Gene Importance Weights Matrix
- GMeans clustering labels
- Integrated gene importance weights
- Training embedding representation

### 4. Integrated Gradients Implementation (`IntegratedGradients.py`)

**Purpose**: Implements the integrated gradients algorithm for interpretability analysis of deep learning models.

**Core features**.
- Keras-compatible implementation
- Multiple output channels support
- Automatic gradient function construction
- Linear interpolation reference points

**Algorithmic Principle**.
The integrated gradient approximates Shapley values by integrating the gradient from the reference input to the actual input, providing interpretability of model decisions.

### 5. Target Composition Generation (`generate_target_combination.py`)

**Role**: Generate and evaluate drug target combinations based on gene importance weights.

**Main Functions**.
- **Data preprocessing**: load and normalise gene importance weights
- **Gene clustering**: group genes using hierarchical clustering
- **Candidate Gene Classification**: Assigns disease-associated genes to gene clusters
- **Target combination generation**: Generate gene pairs based on relevance and significance scores
- **Outcome assessment**: Assess match with known target combinations

**Usage**:
```bash
python generate_target_combination.py
```

**Operational mode**.
- **Predictive mode**: customised parameters, suitable for exploratory analysis
- **Evaluation mode**: Use default parameters, suitable for validating known results

## Data flow

```
1. raw data → PCA preprocessing → deep learning model training
2. trained model → integrated gradient calculation → gene importance weights
3. gene importance weights → GMeans clustering → integrated weights and labels
4. integrated weights → gene clustering → candidate gene classification
5. candidate genes → correlation analysis → target combination generation
6. target combination → known combination matching → prediction effect evaluation
```

## Install dependencies

```bash

pip install numpy pandas scikit-learn matplotlib seaborn scipy


pip install tensorflow  # 或
pip install keras


pip install tensorflow-gpu
```

## Example of use

### Complete process example


```bash

python train_all_models.py \
  --models VAE,AE,DAE \
  --cancer-types A549 \
  --latent-dims 5,10,25,50 \
  --runs 3 \
  --input-root /path/to/input \
  --output-root /path/to/output


python run_interpretability_and_ensembles.py \
  --models VAE \
  --cancer-types A549 \
  --latent-dims 5,10,25,50 \
  --runs 3 \
  --input-root /path/to/input \
  --output-root /path/to/output

python generate_target_combination.py
```

### Parameter configuration description

**Training parameters**.
- `-models`: select model types to be trained
- `-cancer-types`: Specify the cancer types
- `--latent-dims`: latent spatial dimensions
- `--runs`: number of runs per configuration

**Interpretable parameters**.
- `-skip-ig`: skip integrated gradient calculation
- `--pca-method`: feature extraction method (PCA/ICA/RP)

**Target combination parameters**.
- Number of clusters: 2-10 gene clusters
- Number of reference genes: 20-500 genes
- Correlation threshold: 0.0-1.0
- Number of output combinations: 10-500 combinations

### Output file description

### Training output
- `encoder_{dim}L_run{run}.json/h5`: encoder model file
- `latent_{dim}L_run{run}.tsv`: latent representation
- `training_results_{model}_{dims}.json`: training results statistics

### Interpretable output
- `{cancer}_DATA_{model}_Cluster_Weights_TRAINING_{dim}L_fold{run}.tsv`: gene importance weights
- `{cancer}_TRAINING_DATA_kmeans_ENSEMBLE_LABELS_{L}L.txt`: clustering labels
- `{cancer}_DeepProfile_Ensemble_Gene_Importance_Weights_{L}L.tsv`: integration weights
- `{cancer}_DeepProfile_Training_Embedding_{L}L.tsv`: Training Embedding

### Target combination output
- `target_combinations.csv`: predicted target combinations
- `match_statistics.csv`: match statistics
- `gene_cluster_heatmap.svg`: Gene cluster heatmap
- `match_rate_trend.png`: Match rate trend map

## Technical features

1. **Modular design**: Each functional module is independent, easy to maintain and expand.
2. **Multi-model support**: supports multiple deep learning architectures and baseline methods.
3. **Automatic parameter tuning**: GMeans automatically determines the optimal number of clusters.
4. **Interpretability Analysis**: Integrated gradients provide explanations for model decisions.
5. **Results validation**: Match assessment with known target combinations
6. **Visualisation support**: Generate heatmaps, trend charts, etc.

## Application Scenarios

- **Drug discovery**: Predict potential drug target combinations
- **Precision medicine**: personalised treatment strategies based on gene expression patterns
- **Cancer research**: Analyse gene regulatory networks of different cancer types
- **Biomarkers**: identification of disease-associated gene markers
- **Drug Repositioning**: Discovering new indications for existing drugs

## Notes

1. **Data format**: Input data should be in TSV format, including gene expression matrix.
2. **Memory Requirements**: Large scale datasets may require more memory.
3. **GPU support**: GPU acceleration is recommended for deep learning training.
4. **Parameter tuning**: Adjust the clustering and correlation parameters according to specific data characteristics.
5. **Results Interpretation**: Predicted results require biological and clinical validation.



