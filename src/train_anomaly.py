from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from evaluate import Predict
import joblib
import pandas as pd
import os
import numpy as np

class LogisticRegressionModel:

    def train_model(self, X_train, y_train):
        model = LogisticRegression(max_iter=1000)
        model.fit(X_train, y_train)
        print("Logistic regression model's trained !")
        return model


class RandomForestModel:

    def train_model(self, X_train, y_train):
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        print("Random Forest model's trained !")
        return model


class SVMModel:

    def train_model(self, X_train, y_train):
        # probability=True indispensable pour predict_proba et la courbe ROC
        model = SVC(kernel='rbf', random_state=42, probability=True)
        model.fit(X_train, y_train)
        print("SVM model's trained !")
        return model


class IsolationForestModel:

    def train_model(self, X_train):
        model = IsolationForest(n_estimators=100, contamination=0.01, random_state=42)
        model.fit(X_train)
        print("Isolation Forest model's trained !")
        return model


class LocalOutlierFactorModel:

    def train_model(self, X_train):
        # novelty=True obligatoire pour utiliser predict() sur de nouvelles données
        model = LocalOutlierFactor(n_neighbors=20, contamination=0.01, novelty=True)
        model.fit(X_train)
        print("Local Outlier Factor model's trained !")
        return model


class save_model:

    def save_model(self, model, file_path):
        try:
            joblib.dump(model, file_path)
            print(f"Model saved in {file_path}")
        except Exception as e:
            print(f"Error while saving the model : {e}")


if __name__ == "__main__":

    os.makedirs("results", exist_ok=True)
    os.makedirs("figures", exist_ok=True)
    os.makedirs("models/anomaly", exist_ok=True)

    #Loading data                                  

    X_train_anomaly = pd.read_csv("data/processed/X_train_anomaly.csv")
    X_train_scaled  = pd.read_csv("data/processed/X_train_scaled.csv")
    y_train         = pd.read_csv("data/processed/y_train.csv").values.ravel()
    X_test_anomaly  = pd.read_csv("data/processed/X_test_anomaly.csv")
    X_test_scaled   = pd.read_csv("data/processed/X_test_scaled.csv")
    y_test          = pd.read_csv("data/processed/y_test.csv").values.ravel()

    # Training

    model_lr  = LogisticRegressionModel().train_model(X_train_anomaly, y_train)
    model_rf  = RandomForestModel().train_model(X_train_anomaly, y_train)
    model_svm = SVMModel().train_model(X_train_anomaly, y_train)
    model_if  = IsolationForestModel().train_model(X_train_scaled)
    model_lof = LocalOutlierFactorModel().train_model(X_train_scaled)

    #  Prédictions                                                
    predictor = Predict()

    y_pred_lr  = predictor.predict(model_lr,  X_test_anomaly)
    y_pred_rf  = predictor.predict(model_rf,  X_test_anomaly)
    y_pred_svm = predictor.predict(model_svm, X_test_anomaly)

    # Convert -1/1 → 0/1
    y_pred_if  = predictor.predict(model_if,  X_test_scaled)
    y_pred_if  = np.where(y_pred_if  == -1, 1, 0)

    # .values for avoiding warning feature names
    y_pred_lof = predictor.predict(model_lof, X_test_scaled.values)
    y_pred_lof = np.where(y_pred_lof == -1, 1, 0)

    # Metrics
    print("\n--- Logistic Regression metrics ---")
    predictor.evaluate_metrics(y_test, y_pred_lr)

    print("\n--- Random Forest metrics ---")
    predictor.evaluate_metrics(y_test, y_pred_rf)

    print("\n--- SVM metrics ---")
    predictor.evaluate_metrics(y_test, y_pred_svm)

    print("\n--- Isolation Forest metrics ---")
    predictor.evaluate_metrics(y_test, y_pred_if)

    print("\n--- Local Outlier Factor metrics ---")
    predictor.evaluate_metrics(y_test, y_pred_lof)

    #Save predictions in csv
    predictions_df = pd.DataFrame({
        "y_true":                      y_test,
        "y_pred_logistic_regression":  y_pred_lr,
        "y_pred_random_forest":        y_pred_rf,
        "y_pred_svm":                  y_pred_svm,
        "y_pred_isolation_forest":     y_pred_if,
        "y_pred_local_outlier_factor": y_pred_lof
    })
    predictions_df.to_csv("results/predictions_anomaly.csv", index=False)  # typo corrigée
    print("\nPredictions saved → results/predictions_anomaly.csv")

    # ROC Curves
    y_proba_lr  = predictor.predict_proba(model_lr,  X_test_anomaly)
    y_proba_rf  = predictor.predict_proba(model_rf,  X_test_anomaly)
    y_proba_svm = predictor.predict_proba(model_svm, X_test_anomaly)

    # IF : inverser le signe car score négatif = anomalie
    y_proba_if  = -model_if.decision_function(X_test_scaled)

    # LOF 
    y_proba_lof = -model_lof.decision_function(X_test_scaled.values)

    predictor.plot_roc_curve("Logistic Regression - Anomaly", y_test, y_proba_lr,  "figures/roc_lr_anomaly.png")
    predictor.plot_roc_curve("Random Forest - Anomaly",       y_test, y_proba_rf,  "figures/roc_rf_anomaly.png")
    predictor.plot_roc_curve("SVM - Anomaly",                 y_test, y_proba_svm, "figures/roc_svm_anomaly.png")
    predictor.plot_roc_curve("Isolation Forest - Anomaly",    y_test, y_proba_if,  "figures/roc_if_anomaly.png")
    predictor.plot_roc_curve("Local Outlier Factor - Anomaly",y_test, y_proba_lof, "figures/roc_lof_anomaly.png")

    #Saving models
    save = save_model()
    save.save_model(model_lr,  "models/anomaly/logistic_regression_anomaly.pkl")
    save.save_model(model_rf,  "models/anomaly/random_forest_anomaly.pkl")
    save.save_model(model_svm, "models/anomaly/svm_anomaly.pkl")
    save.save_model(model_if,  "models/anomaly/isolation_forest_anomaly.pkl")
    save.save_model(model_lof, "models/anomaly/local_outlier_factor_anomaly.pkl")
    print("\nSaving done !")