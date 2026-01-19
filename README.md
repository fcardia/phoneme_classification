# Phoneme Classification
A comprehensive data mining project for classifying audio signals as **nasal sounds** or **oral vowels** using various machine learning algorithms.

[![Python](https://img.shields.io/badge/Python-3.11.4-blue.svg)](https://www.python.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.8.0-9cf.svg)](https://scikit-learn.org/)
[![matplotlib](https://img.shields.io/badge/matplotlib-3.10.8-ff6f61.svg)](https://matplotlib.org/)
[![seaborn](https://img.shields.io/badge/seaborn-0.13.2-4b8bbe.svg)](https://seaborn.pydata.org/)
[![xgboost](https://img.shields.io/badge/xgboost-3.1.2-f0ad4e.svg)](https://xgboost.ai/)
[![tensorflow](https://img.shields.io/badge/tensorflow-2.20.0-ff6a00.svg)](https://www.tensorflow.org/)
[![numpy](https://img.shields.io/badge/numpy-2.3.0-013243.svg)](https://numpy.org/)



---
This project applies data mining techniques to the [Phoneme dataset](https://openml.org/search?type=data&id=1489) from OpenML, aiming to accurately classify sounds into two categories:
- **Class 0**: Nasal sounds
- **Class 1**: Oral vowels

## Objectives

- Perform exploratory data analysis on audio signal features
- Handle class imbalance and preprocess data effectively
- Compare multiple classification algorithms
- Evaluate the impact of dimensionality reduction on model performance

## Models & Algorithms

### Classification Models
| Model            | Description                 |
|------------------|-----------------------------|
| RandomForest     | Tree ensemble averaging     |
| CustomEnsemble   | User-defined ensemble       |
| XGBoost          | Gradient boosted trees      |
| KNN              | Nearest neighbors voting    |
| SVC              | Maximum margin classifier   |
| DecisionTree     | Single decision tree        |
| SVC_UMAP         | SVC with UMAP               |
| MLP              | Feedforward neural network |
| DeepConvModel    | Convolutional deep model    |
| AdaBoost         | Adaptive boosting ensemble |
| SVC_PCA          | SVC with PCA                |
| NaiveBayes       | Probabilistic Bayesian model|


### Dimensionality Reduction
- Feature extraction techniques are used like PCA, LDA, t-SNE, UMAP


## Getting Started

### Prerequisites
The Python version we tested to work for the project is **3.11.4**

To install all the necessary dependencies, create a virtual environment named PHONEME and install the requirements.

In **macOS/Linux** run
```bash
python -m venv PHONEME
source PHONEME/bin/activate

pip install -r requirements.txt

```
In **Windows** run
```bash
python -m venv PHONEME
PHONEME\Scripts\activate

pip install -r requirements.txt

```

## Folder organization
```
phoneme_classification/
├── data/     # sets for validation
├── results/  # model performances
├── hyperparameter_tuning.ipynb
├── main.ipynb
├── README.md
├── requirements.txt
└── tuned_hyperparameters.json
```
- `data/` contains the train sets that will be used for validation and hyperparameter tuning
- `results/` contains the insights about the model performances
- `hyperparameter_tuning.ipynb` contains the validation phase (cross validation and hyperparameter tuning), our suggestion is NOT to run it because it can take more than 1 hour and the optimal hyperparameters are already been found and saved in `tuned_hyperparameters.json`, however if you want to run it, follow these steps:
  1. If you do not run the notebook or interrupt the run before it ends, you don't need to do any other modification.
  2. If you let the tuning run till the end, the new hyperparameters will be saved in `tuned_hyperparameters_new.json` (they will be the same as the existing ones because of the random seed, unless something in the configuration is changed). To use them, in `main.ipynb` change the `HYPERPARAMETERS_PATH` value to `"tuned_hyperparameters_new.json"`.
- `main.ipynb` is the core file of the project and contains our experiment and findings for the phoneme classification problem

## Workflow

### 1) Data Analysis
- Dataset overview and statistics
- Class and feature distribution analysis
- Feature visualization (2D plots, parallel coordinates)

### 2) Preprocessing
- Missing value handling
- Train-test split
- Class balancing
- Feature Extraction

### 3) Classification
- Creation of two custom models
- Model training with optimized hyperparameters
- Model testing
- Performance plots rendering

### 4) Evaluation
- RQ1: Model performances
- RQ2: Class balance impact
- Comparative plots and tables

### 5) Discussion
- Q1: Best performers
- Q2: Impact of dimensionality reduction
- Q3: Sensitivity to dimensionality reduction
- Q4: Strengths, Limitations, Extensions
 
---


<p align="center">
  Cardia F. Loddo M. N.<br>
  Data Mining Project 2025/2026<br>
  CdLM in Computer Science - Applied Artificial Intelligence<br>
  Cagliari State University
</p>
