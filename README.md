# SafeFlow AI — Anti-Money Laundering Detection

> **CSCI323 Group Project** | Machine Learning-based Financial Transaction Monitoring

SafeFlow AI is a machine learning pipeline designed to detect suspicious financial transactions indicative of money laundering. Built on a large-scale synthetic AML dataset, the project covers the full ML workflow: data ingestion, feature engineering, anomaly detection, preprocessing, model training, hyperparameter tuning, and evaluation.

---

## Table of Contents

- [Overview](#overview)
- [Dataset](#dataset)
- [Project Structure](#project-structure)
- [Pipeline Walkthrough](#pipeline-walkthrough)
- [Models](#models)
- [Results](#results)
- [Requirements](#requirements)
- [Usage](#usage)

---

## Overview

Financial institutions face growing pressure to detect money laundering in real time. SafeFlow AI addresses this challenge by:

- Extracting temporal features from raw transaction timestamps
- Detecting anomalies using Isolation Forest
- Handling severe class imbalance with SMOTE
- Training and tuning four classifiers, selecting the best performer by F1 score

---

## Dataset

**Source:** [Synthetic Transaction Monitoring Dataset (SAML-D)](https://www.kaggle.com/datasets/berkanoztas/synthetic-transaction-monitoring-dataset-aml) via KaggleHub

The dataset contains over **9 million synthetic transactions** with features including:

| Feature | Description |
|---|---|
| `Date`, `Time` | Transaction timestamp |
| `Amount` | Transaction value |
| `Payment_type` | Method of payment |
| `Payment_currency` | Currency used |
| `Sender_account` | Sending account identifier |
| `Receiver_account` | Receiving account identifier |
| `Is_laundering` | Binary target label (0 = normal, 1 = suspicious) |
| `Laundering_type` | Type of laundering scheme (dropped to prevent leakage) |

---

## Project Structure

```
SafeFlowAI_CodeBase_CSCI323.ipynb   # Main notebook
best_model_ad.joblib                # Saved AdaBoost model (generated after training)
README.md
```

---

## Pipeline Walkthrough

### 1. Data Ingestion
Downloads the SAML-D dataset via `kagglehub` and loads it into a Pandas DataFrame.

### 2. Feature Engineering
Extracts temporal features from the raw timestamp column:
- `hour_` — hour of day (captures unusual transaction times)
- `day_` — day of week (flags weekend activity)
- `month_` — month of year (detects seasonal patterns)

### 3. Anomaly Detection
Uses **Isolation Forest** (contamination=1%) to generate an `anomaly_score` for each transaction. Lower scores indicate more anomalous behavior. Results are visualized against Amount, Hour, Month, and Payment Type.

### 4. Correlation Analysis
A heatmap of the correlation matrix confirms no multicollinearity among features, ensuring no redundant information and no restriction on model selection.

### 5. Data Partitioning
An 80/20 stratified train/test split is applied **before preprocessing** to prevent data leakage.

### 6. Preprocessing
A `ColumnTransformer` pipeline applies:
- **StandardScaler** for numerical features
- **OneHotEncoder** (drop first, ignore unknown) for categorical features

### 7. Low-Variance Feature Removal
`VarianceThreshold(0.01)` removes features where a single value appears in more than 99% of rows, reducing noise.

### 8. Class Imbalance Handling
**SMOTE** (Synthetic Minority Oversampling Technique) is applied to the training set to produce a balanced class distribution before model training.

---

## Models

Four classifiers were selected for their scalability with large tabular datasets:

| Model | Rationale |
|---|---|
| **Logistic Regression** | Fast, interpretable baseline; `saga` solver for large data |
| **Random Forest** | Ensemble method; captures complex feature interactions |
| **AdaBoost** | Adaptive boosting; learns from misclassifications iteratively |
| **Gaussian Naive Bayes** | Probabilistic; extremely fast on large datasets |

### Hyperparameter Tuning

- **GridSearchCV** — used for Logistic Regression and AdaBoost
- **RandomizedSearchCV** — used for Random Forest (more efficient over large parameter spaces)
- Tuning is performed on a **stratified subset (~2% of resampled data)** using `StratifiedShuffleSplit` for computational efficiency

---

## Results

Models are evaluated on the held-out test set using Precision, Recall, F1 Score, ROC AUC, and Confusion Matrix.

**F1 Score is the primary metric** due to the imbalanced nature of the dataset.

> **AdaBoost was selected as the final model**, achieving the highest F1 Score and Precision among all four candidates, representing the best balance between correctly identifying laundering transactions and minimizing false alarms.

The trained model is saved to disk using `joblib`:

```python
joblib.dump(best_ad, 'best_model_ad.joblib')
```

---

## Requirements

Install dependencies via pip:

```bash
pip install kagglehub pandas numpy matplotlib seaborn scikit-learn imbalanced-learn joblib
```

> **Note:** A Kaggle account and API credentials are required to download the dataset via `kagglehub`. Set up your `kaggle.json` credentials before running.

---

## Usage

1. Clone the repository and install dependencies.
2. Configure your Kaggle API credentials.
3. Open and run `SafeFlowAI_CodeBase_CSCI323.ipynb` cell by cell in Jupyter Notebook or JupyterLab.
4. After training, the best model (`best_model_ad.joblib`) will be saved to your working directory.
5. The final cells demonstrate sample prediction on a randomly selected positive (laundering) transaction from the test set, printing transaction details when a correct prediction is made.

---

## Notes

- `Laundering_type` is dropped before training to prevent **label leakage**.
- `Sender_account` and `Receiver_account` are treated as identifiers and excluded from features.
- `GaussianNB` requires dense array input; sparse matrices are converted with `.toarray()` before prediction.
