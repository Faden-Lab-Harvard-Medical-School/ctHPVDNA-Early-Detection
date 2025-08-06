# HPV+ Cancer Prediction

This repo contains trained machine learning models to predict HPV-positive (HPV+) cancer from healthy and cancer control samples using features generated from **HPV-DeepSeek**. The models have been trained and saved for direct reuse in downstream tasks such as evaluation, validation, and deployment.

---

## 📁 Contents

| File | Description |
|------|-------------|
| `ML Training.ipynb` | Jupyter notebook for training, evaluating, and saving machine learning models using HPV-DeepSeek input features. |
| `RandomForest_model.pkl` | Trained Random Forest classifier for HPV+ cancer prediction. |
| `XGBoost_model.pkl` | Trained XGBoost classifier. |
| `AdaBoost_model.pkl` | Trained AdaBoost classifier. |
| `NaiveBayes_model.pkl` | Trained Gaussian Naive Bayes classifier. |
| `README.md` | Project overview and usage instructions. |

---

## 🔍 Model Overview

Models were trained using tabular data output from the HPV-DeepSeek pipeline. The target label represents three classes:
- HPV-positive cancer
- Healthy controls
- Cancer controls (HPV-negative)

### Models included:
- **RandomForestClassifier** (scikit-learn)
- **XGBoostClassifier** (XGBoost)
- **AdaBoostClassifier** (scikit-learn)
- **GaussianNB** (scikit-learn)

Each model was trained on a stratified training set and validated on a held-out test set using standard classification metrics (accuracy, precision, recall, F1 score, AUC).

---

## 🧠 How to Use the Trained Models

### 1. Install dependencies

```bash
pip install scikit-learn xgboost joblib pandas
```
### 2. Load a model

import joblib

```python
rf_model = joblib.load('RandomForest_model.pkl')
xgb_model = joblib.load('XGBoost_model.pkl')
ada_model = joblib.load('AdaBoost_model.pkl')
nb_model = joblib.load('NaiveBayes_model.pkl')
```

### 3. Use for prediction

```python

# Assuming you have a new sample in the same feature format

import pandas as pd

new_data = pd.read_csv('new_input.csv')  # Replace with your actual input
predictions = rf_model.predict(new_data)

```
