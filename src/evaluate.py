from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix, roc_auc_score, roc_curve
)
import matplotlib.pyplot as plt
import pandas as pd
import joblib


class Predict:

    def load_model(self, file_path):
        try:
            model = joblib.load(file_path)
            print(f"Modèle chargé depuis {file_path}")
            return model
        except Exception as e:
            print(f"Erreur lors du chargement du modèle : {e}")
            return None

    def predict(self, model, X):
        try:
            predictions = model.predict(X)
            print("Prédictions effectuées.")
            return predictions
        except Exception as e:
            print(f"Erreur lors de la prédiction : {e}")
            return None

    def predict_proba(self, model, X):
        """
        Retourne les probabilités de la classe positive (fraude).
        Nécessaire pour calculer la courbe ROC.
        Fonctionne avec LR, RF — pour SVM utiliser decision_function.
        """
        try:
            if hasattr(model, "predict_proba"):
                proba = model.predict_proba(X)[:, 1]
            elif hasattr(model, "decision_function"):
                proba = model.decision_function(X)
            else:
                raise ValueError("Le modèle ne supporte ni predict_proba ni decision_function.")
            return proba
        except Exception as e:
            print(f"Erreur lors du calcul des probabilités : {e}")
            return None

    def evaluate_metrics(self, y_true, y_pred):
        try:
            accuracy  = accuracy_score(y_true, y_pred)
            precision = precision_score(y_true, y_pred)
            recall    = recall_score(y_true, y_pred)
            f1        = f1_score(y_true, y_pred)
            cm        = confusion_matrix(y_true, y_pred)

            print(f"Accuracy  : {accuracy:.4f}")
            print(f"Precision : {precision:.4f}")
            print(f"Recall    : {recall:.4f}")
            print(f"F1 Score  : {f1:.4f}")
            print(f"Confusion Matrix:\n{cm}")

        except Exception as e:
            print(f"Erreur lors de l'évaluation des métriques : {e}")

    def plot_roc_curve(self, model_name, y_true, y_proba, save_path=None):
        try:
            fpr, tpr, _ = roc_curve(y_true, y_proba)
            auc_score   = roc_auc_score(y_true, y_proba)

            plt.figure(figsize=(8, 6))
            plt.plot(fpr, tpr, color="darkorange", lw=2,
                     label=f"ROC curve (AUC = {auc_score:.4f})")
            plt.plot([0, 1], [0, 1], color="navy", lw=1.5,
                     linestyle="--", label="Random classifier")

            plt.xlabel("False Positive Rate")
            plt.ylabel("True Positive Rate")
            plt.title(f"Courbe ROC — {model_name}")
            plt.legend(loc="lower right")
            plt.tight_layout()

            if save_path:
                plt.savefig(save_path, dpi=150)
                print(f"Courbe ROC sauvegardée → {save_path}")

            plt.show()
            print(f"AUC Score : {auc_score:.4f}")
            return auc_score

        except Exception as e:
            print(f"Erreur lors du tracé de la courbe ROC : {e}")
            return None

    def save_predictions(self, predictions, file_path):
        try:
            pd.DataFrame(predictions, columns=["Predictions"]).to_csv(file_path, index=False)
            print(f"Prédictions sauvegardées dans {file_path}")
        except Exception as e:
            print(f"Erreur lors de la sauvegarde des prédictions : {e}")


# ------------------------------------------------------------------ #
#  PIPELINE EVALUATION                                                #
# ------------------------------------------------------------------ #

if __name__ == "__main__":

    predictor = Predict()

    # Loading test data
    X_test = pd.read_csv("data/processed/X_test.csv")
    X_test_scaled = pd.read_csv("data/processed/X_test_scaled.csv")
    X_test_anomaly = pd.read_csv("data/processed/X_test_anomaly.csv")
    y_test = pd.read_csv("data/processed/y_test.csv").values.ravel()

    # Load models for class_weights approach
    model_lr = predictor.load_model("models/class_weights/logistic_regression_model.pkl")
    model_rf = predictor.load_model("models/class_weights/random_forest_model.pkl")
    model_svm = predictor.load_model("models/class_weights/svm_model.pkl")

    predictions_random_forest = predictor.predict(model_rf, X_test)
    predictions_svm = predictor.predict(model_svm, X_test)
    predictions_logistic_regression = predictor.predict(model_lr, X_test_scaled)

    #Evaluate me
    print("Métriques du modèle Random Forest :")
    predictor.evaluate_metrics(y_test, predictions_random_forest)
    print("Métriques du modèle SVM :")
    predictor.evaluate_metrics(y_test, predictions_svm)
    print("Métriques du modèle Logistic Regression :")
    predictor.evaluate_metrics(y_test, predictions_logistic_regression)

