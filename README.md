# 💳 Credit Card Fraud Detection

A machine learning project for detecting fraudulent credit card transactions using multiple class imbalance handling strategies and a diverse set of models.

---

## 📌 Project Overview

Credit card fraud detection is a classic **highly imbalanced classification problem** — in this dataset, only **~1% of transactions are fraudulent** out of 284,807 transactions. This makes it critical to go beyond simple accuracy and focus on metrics like **Recall** (catching actual fraud) and **Precision** (avoiding false alarms).

This project explores and compares **three different strategies** to handle class imbalance, each applied to a set of supervised and unsupervised models.

---

## 📁 Project Structure

```
Credit-Card-Fault-Detection/
│
├── data/
│   ├── raw/                        # Original dataset (creditcard.csv)
│   └── processed/                  # Preprocessed datasets
│       ├── X_train_scaled.csv      # Scaled only → for IF, LOF
│       ├── X_train_smote.csv       # Scaled + SMOTE → for LR, RF, SVM
│       ├── X_train_anomaly.csv     # Scaled + anomaly_score feature → for LR, RF, SVM
│       ├── X_test_scaled.csv       # Test set (never modified)
│       ├── X_test_anomaly.csv      # Test set with anomaly_score column
        ├── X_test.csv
│       ├── y_train.csv
│       ├── y_train_smote.csv
│       └── y_test.csv
│
├── models/
│   ├── class_weights/              # LR, RF, SVM trained with class_weight parameter
│   ├── smote/                      # LR, RF, SVM trained on SMOTE-resampled data
│   └── anomaly/                    # LR, RF, SVM trained on anomaly-enriched data
│                                   # + Isolation Forest and LOF
│
├── figures/                        # ROC curves for all models and approaches
├── results/                        # Prediction CSVs for all approaches
│
├── src/
│   ├── preprocessing.py            # Full preprocessing pipeline
│   ├── train.py                    # Training pipeline — Class Weights approach
│   ├── train_smote.py              # Training pipeline — SMOTE approach
│   ├── train_anomaly.py            # Training pipeline — Anomaly approach
│   └── evaluate.py                 # Metrics, ROC curve, prediction utilities
│
└── README.md
```

---

## 📊 Dataset

- **Source**: [Kaggle — Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)
- **Size**: 284,807 transactions
- **Fraud rate**: ~0.17% (highly imbalanced)
- **Features**: `V1` to `V28` (PCA-anonymized) + `Amount`
- **Target**: `Class` (0 = normal, 1 = fraud)
- **Note**: The `Time` column was dropped as it represents raw elapsed seconds and adds no direct predictive value at this stage of the project.

---

## ⚙️ Preprocessing Pipeline

All preprocessing is handled in `preprocessing.py` through the `DataPreprocessing` class.

**Steps applied:**
1. Load data and remove duplicates
2. Drop irrelevant features (`Time`)
3. Separate features `X` and target `y`
4. Stratified train/test split (80/20) — stratification is critical with imbalanced classes
5. **RobustScaler** — chosen over StandardScaler because it is resistant to outliers (extreme transaction amounts), and over MinMaxScaler for the same reason
6. Apply the three imbalance strategies to generate three versions of the training set

**Three versions of the training set are generated:**

| Dataset | Scaling | SMOTE | Anomaly Score | Used for |
|---|---|---|---|---|
| `X_train_scaled` | ✅ | ❌ | ❌ | Isolation Forest, LOF |
| `X_train_smote` | ✅ | ✅ | ❌ | LR, RF, SVM (SMOTE approach) |
| `X_train_anomaly` | ✅ | ❌ | ✅ | LR, RF, SVM (Anomaly approach) |

> The test set is **never modified** — `X_test_scaled` is used for all approaches except Anomaly, which uses `X_test_anomaly` (same test data, with the `anomaly_score` column added for consistency with the trained model).

---

## 🔁 Class Imbalance Strategies

### 1. Class Weights
No data modification. The imbalance is handled inside the model's loss function by assigning a higher penalty to misclassified fraud cases. Applied via `class_weight='balanced'` in sklearn models.

### 2. SMOTE (Synthetic Minority Oversampling Technique)
Generates synthetic fraud samples to rebalance the training set. `SMOTETomek` was used — a combination of SMOTE oversampling and Tomek Links cleaning, which removes ambiguous borderline samples after resampling for a cleaner decision boundary.

### 3. Anomaly Score Feature (Isolation Forest)
An `IsolationForest` is trained on normal transactions only and used to compute an `anomaly_score` for every transaction. This score is added as an extra feature to the training and test sets. The data remains imbalanced — this approach enriches the feature space rather than rebalancing it, and can be combined with `class_weights` for stronger results.

---

## 🤖 Models

| Model | Type | Approach |
|---|---|---|
| Logistic Regression | Supervised | Class Weights, SMOTE, Anomaly |
| Random Forest | Supervised | Class Weights, SMOTE, Anomaly |
| SVM (RBF kernel) | Supervised | Class Weights, SMOTE, Anomaly |
| Isolation Forest | Unsupervised | Anomaly (trained on `X_train_scaled`) |
| Local Outlier Factor | Unsupervised | Anomaly (trained on `X_train_scaled`) |

---

## 📈 Results

### Class Weights Approach

| Model | Accuracy | Precision | Recall | F1 Score | AUC |
|---|---|---|---|---|---|
| Logistic Regression | 0.9772 | 0.0607 | 0.8737 | 0.1135 | 0.9669 |
| Random Forest | 0.9995 | 0.9710 | 0.7053 | 0.8171 | — |
| SVM | 0.9930 | 0.1492 | 0.6737 | 0.2443 | 0.9596 |

### SMOTE Approach

| Model | Accuracy | Precision | Recall | F1 Score | AUC |
|---|---|---|---|---|---|
| Logistic Regression | 0.9763 | 0.0587 | 0.8737 | 0.1099 | 0.9627 |
| Random Forest | 0.9994 | 0.8795 | 0.7684 | 0.8202 | 0.9549 |
| SVM | 0.9861 | 0.0921 | 0.8211 | 0.1656 | 0.9399 |

### Anomaly Score Approach

| Model | Accuracy | Precision | Recall | F1 Score | AUC |
|---|---|---|---|---|---|
| Logistic Regression | 0.9992 | 0.8636 | 0.6000 | 0.7081 | 0.9568 |
| Random Forest | 0.9995 | 0.9722 | 0.7368 | 0.8383 | 0.9245 |
| SVM | 0.9993 | 0.9831 | 0.6105 | 0.7532 | 0.8866 |
| Isolation Forest | 0.9901 | 0.1034 | 0.6421 | 0.1781 | 0.9396 |
| Local Outlier Factor | 0.9873 | 0.0032 | 0.0211 | 0.0055 | 0.5766 |

---

## 🏆 Best Model

**Random Forest — Anomaly Score Approach** is the best overall model:

```
Precision : 0.9722  →  when it predicts fraud, it is right 97% of the time
Recall    : 0.7368  →  it catches 74% of all actual frauds
F1 Score  : 0.8383  →  best precision/recall balance across all experiments
```

It consistently outperforms other models across all three approaches, combining high precision with a solid recall — the most important trade-off in fraud detection, where missing a fraud is far more costly than triggering a false alert.

**Notable observations:**
- **Logistic Regression and SVM** suffer from very low Precision under Class Weights and SMOTE, generating many false positives. The Anomaly Score approach dramatically fixes this (LR Precision: 0.06 → 0.86, SVM: 0.15 → 0.98) but at the cost of lower Recall.
- **Isolation Forest** performs reasonably well as a standalone unsupervised detector (AUC: 0.94) with no labels required during training.
- **LOF** performs poorly on this dataset (AUC: 0.58), which is expected — it suffers from the **curse of dimensionality** with 30 features, as distance-based methods lose effectiveness in high-dimensional spaces.
- **SMOTE did not significantly improve** over Class Weights on this dataset — results are very similar, suggesting the PCA-transformed features (`V1`–`V28`) are already well-structured enough that synthetic oversampling adds little value.

---

## 🔮 Perspectives

- **Temporal feature engineering**: The `Time` column could be transformed into `hour_of_day` and `is_night` features to capture fraud patterns at unusual hours.
- **Threshold tuning**: Adjusting the classification threshold (default 0.5) could further improve the Recall/Precision trade-off depending on business requirements.
- **Ensemble methods**: Combining the anomaly score from Isolation Forest with SMOTE resampling could yield stronger results.
- **XGBoost / LightGBM**: Gradient boosting models are known to perform well on fraud detection and would be a natural next step.

---

## 🛠️ Installation

```bash
git clone https://github.com/KANOHA242/Credit-Card-Fault-Detection.git
cd Credit-Card-Fault-Detection
pip install -r requirements.txt
```

**Run the full pipeline:**
```bash
# 1 — Preprocessing (generates all dataset versions)
python src/preprocessing.py

# 2 — Training (one file per approach)
python src/train.py            # Class Weights
python src/train_smote.py      # SMOTE
python src/train_anomaly.py    # Anomaly Score
```

---

## 📦 Requirements

```
pandas
numpy
scikit-learn
imbalanced-learn
matplotlib
joblib
```

---

## 👩‍💻 Author
"KANOHA ELENGA Helmie Naella Jihane"

