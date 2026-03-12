from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from preprocessing import DataPreprocessing
from evaluate import Predict
import joblib
import pandas as pd
import os


class LogisticRegressionModel:

    def train_model(self, X_train, y_train):
        model = LogisticRegression(class_weight='balanced', max_iter=1000)
        model.fit(X_train, y_train)
        print("Logistic Regression trained!")
        return model


class RandomForestModel:

    def train_model(self, X_train, y_train):
        model = RandomForestClassifier(class_weight='balanced', n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        print("Random Forest trained!")
        return model


class SVMModel:

    def train_model(self, X_train, y_train):
        model = SVC(kernel='rbf', random_state=42, class_weight='balanced', probability=True)
        model.fit(X_train, y_train)
        print("SVM trained!")
        return model


class save_model:

    def save_model(self, model, file_path):
        try:
            joblib.dump(model, file_path)
            print(f"Modèle sauvegardé → {file_path}")
        except Exception as e:
            print(f"Erreur lors de la sauvegarde : {e}")


if __name__ == "__main__":

    os.makedirs("results", exist_ok=True)
    #Loading data
    X_train_scaled = pd.read_csv("data/processed/X_train_scaled.csv")
    y_train        = pd.read_csv("data/processed/y_train.csv").values.ravel()
    X_test_scaled  = pd.read_csv("data/processed/X_test_scaled.csv")
    y_test         = pd.read_csv("data/processed/y_test.csv").values.ravel()

    #Training models
    model_lr  = LogisticRegressionModel().train_model(X_train_scaled, y_train)
    model_rf  = RandomForestModel().train_model(X_train_scaled, y_train)
    model_svm = SVMModel().train_model(X_train_scaled, y_train)

    #Predictions + metrics
    predictor = Predict()

    y_pred_lr  = predictor.predict(model_lr,  X_test_scaled)
    y_pred_rf  = predictor.predict(model_rf,  X_test_scaled)
    y_pred_svm = predictor.predict(model_svm, X_test_scaled)

    print("\n--- Logistic Regression metrics ---")
    predictor.evaluate_metrics(y_test, y_pred_lr)

    print("\n--- Random Forest metrics ---")
    predictor.evaluate_metrics(y_test, y_pred_rf)

    print("\n--- SVM metrics ---")
    predictor.evaluate_metrics(y_test, y_pred_svm)

    #Saving predictions
    predictions_df = pd.DataFrame({
        "y_true":                y_test,
        "y_pred_logistic_regression": y_pred_lr,
        "y_pred_random_forest":       y_pred_rf,
        "y_pred_svm":                 y_pred_svm
    })
    predictions_df.to_csv("results/predictions_class_weights.csv", index=False)
    print("\nPrédictions sauvegardées → results/predictions_class_weights.csv")

    # ROC Curves
    y_proba_lr  = predictor.predict_proba(model_lr,  X_test_scaled)
    y_proba_rf  = predictor.predict_proba(model_rf,  X_test_scaled)
    y_proba_svm = predictor.predict_proba(model_svm, X_test_scaled)

    predictor.plot_roc_curve(
        model_name="Logistic Regression - Class Weights",
        y_true=y_test,
        y_proba=y_proba_lr,
        save_path="figures/roc_lr_class_weights.png"
    )
    predictor.plot_roc_curve(
        model_name="Random Forest - Class Weights",
        y_true=y_test,
        y_proba=y_proba_rf,
        save_path="figures/roc_rf_class_weights.png"
    )
    predictor.plot_roc_curve(
        model_name="SVM - Class Weights",
        y_true=y_test,
        y_proba=y_proba_svm,
        save_path="figures/roc_svm_class_weights.png"
    )

    #Saving models
    save = save_model()
    save.save_model(model_lr,  "models/class_weights/logistic_regression_cw.pkl")
    save.save_model(model_rf,  "models/class_weights/random_forest_cw.pkl")
    save.save_model(model_svm, "models/class_weights/svm_cw.pkl")
    print("\nSauvegarde des modèles terminée.")