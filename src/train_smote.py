from sklearn.ensemble import RandomForestClassifier
from preprocessing import DataPreprocessing
from evaluate import Predict
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
import joblib
import pandas as pd
import os

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

        model = SVC(kernel='rbf', random_state=42)
        model.fit(X_train, y_train)

        print("SVM model's trained !")
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
    #Loading data
    X_train_smote = pd.read_csv("data/processed/X_train_smote.csv")
    y_train_smote = pd.read_csv("data/processed/y_train_smote.csv").values.ravel()
    X_test_scaled  = pd.read_csv("data/processed/X_test_scaled.csv")
    y_test         = pd.read_csv("data/processed/y_test.csv").values.ravel()

    #Training models
    model_lr  = LogisticRegressionModel().train_model(X_train_smote, y_train_smote)
    model_rf  = RandomForestModel().train_model(X_train_smote, y_train_smote)
    model_svm = SVMModel().train_model(X_train_smote, y_train_smote)

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
    predictions_df.to_csv("results/predictions_smote.csv", index=False)
    print("\nPredictions saved → results/predictions_smote.csv")

    # ROC Curves
    y_proba_lr  = predictor.predict_proba(model_lr,  X_test_scaled)
    y_proba_rf  = predictor.predict_proba(model_rf,  X_test_scaled)
    y_proba_svm = predictor.predict_proba(model_svm, X_test_scaled)

    predictor.plot_roc_curve(
        model_name="Logistic Regression - SMOTE",
        y_true=y_test,
        y_proba=y_proba_lr,
        save_path="figures/roc_lr_smote.png"
    )
    predictor.plot_roc_curve(
        model_name="Random Forest - SMOTE",
        y_true=y_test,
        y_proba=y_proba_rf,
        save_path="figures/roc_rf_smote.png"
    )
    predictor.plot_roc_curve(
        model_name="SVM - SMOTE",
        y_true=y_test,
        y_proba=y_proba_svm,
        save_path="figures/roc_svm_smote.png"
    )

    #Saving models
    save = save_model()
    save.save_model(model_lr,  "models/smote/logistic_regression_smote.pkl")
    save.save_model(model_rf,  "models/smote/random_forest_smote.pkl")
    save.save_model(model_svm, "models/smote/svm_smote.pkl")
    print("\n Saving done !")