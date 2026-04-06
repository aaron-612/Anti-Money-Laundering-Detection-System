# Anti-Money Laundering Detection System

A machine learning-based system for detecting suspicious financial transactions using a combination of anomaly detection and supervised classification models. This project focuses on identifying potential money laundering activities in large-scale transaction data.

## Overview

Financial fraud detection is a highly imbalanced and complex problem. This project uses a hybrid approach:

- Unsupervised learning to detect unusual transaction patterns  
- Supervised learning to classify transactions as legitimate or suspicious  
- Imbalanced data handling to improve detection of rare fraud cases  

## Dataset

Source: Synthetic AML dataset from Kaggle

The dataset contains millions of transaction records with:
- Transaction amount  
- Sender and receiver accounts  
- Payment type and currency  
- Date and time  
- Fraud label (Is_laundering)  

## Key Features

### Feature Engineering
- Created time-based features:
  - hour_
  - day_
  - month_
- Combined date and time into a timestamp  
- Removed leakage-prone features  

### Anomaly Detection
- Implemented Isolation Forest  
- Generated anomaly_score for each transaction  
- Used to identify unusual transaction behavior  

### Data Preprocessing
- Standardization using StandardScaler  
- One-hot encoding for categorical features  
- Low variance feature removal  
- Train test split with stratification  

### Handling Imbalanced Data
- Applied SMOTE (Synthetic Minority Oversampling Technique)  
- Balanced fraud and non-fraud samples  

### Models Used
- Logistic Regression  
- Random Forest  
- AdaBoost (best performing model)  
- Gaussian Naive Bayes  

### Hyperparameter Tuning
- GridSearchCV for Logistic Regression and AdaBoost  
- RandomizedSearchCV for Random Forest  
- Optimized using F1 score for imbalanced data  

## Evaluation Metrics

- Precision  
- Recall  
- F1 Score  
- ROC AUC  
- Confusion Matrix  

## Results

- Best model: AdaBoost Classifier  
- Achieved highest F1 score among tested models  
- Provided best balance between precision and recall  

## Sample Prediction

The model can:
- Identify suspicious transactions  
- Output prediction (fraud or non-fraud)  
- Display transaction details for flagged cases  

## Model Saving

```python
joblib.dump(best_ad, 'best_model_ad.joblib')

Load model:

joblib.load('best_model_ad.joblib')
Project Structure
aml-detection/
│── aml_detection.ipynb
│── best_model_ad.joblib
│── README.md
Tech Stack
Python
Pandas
NumPy
Scikit-learn
Imbalanced-learn
Matplotlib
Seaborn
How to Run
Install dependencies:
pip install pandas numpy scikit-learn imbalanced-learn matplotlib seaborn kagglehub
Run the notebook:
jupyter notebook aml_detection.ipynb
Key Learnings
Handling highly imbalanced datasets
Combining unsupervised and supervised learning
Feature engineering for time-based data
Model tuning for large datasets
Future Improvements
Deploy as a real-time fraud detection API
Use deep learning models such as LSTM or autoencoders
Integrate streaming pipelines using Kafka and Spark
Add explainability using SHAP or LIME
