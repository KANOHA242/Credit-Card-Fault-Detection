import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler
from imblearn.over_sampling import SMOTE, ADASYN
from imblearn.under_sampling import RandomUnderSampler
from imblearn.combine import SMOTETomek
from sklearn.ensemble import IsolationForest


class DataPreprocessing:

    def load_data(self, file_path):
        try:
            data = pd.read_csv(file_path)
            print(f"Données chargées : {data.shape}")
            return data
        except Exception as e:
            print(f"Erreur chargement : {e}")
            return None

    def remove_duplicates(self, data):
        initial = data.shape[0]
        data = data.drop_duplicates()
        print(f"Doublons supprimés : {initial - data.shape[0]}")
        return data

    def delete_features(self, data, features_to_delete):
        data = data.drop(columns=[col for col in features_to_delete if col in data.columns])
        print(f"Features supprimées : {features_to_delete}")
        return data

    def separation(self, data, target_col="Class"):
        X = data.drop(target_col, axis=1)
        y = data[target_col].values.ravel()
        fraud_rate = y.mean() * 100
        print(f"Séparation OK | Taux de fraude : {fraud_rate:.2f}%")
        return X, y

    def split_data(self, X, y, test_size=0.2, random_state=42):
        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=test_size,
            random_state=random_state,
            stratify=y      # crucial avec classes déséquilibrées
        )
        print(f"Train : {X_train.shape} | Test : {X_test.shape}")
        return X_train, X_test, y_train, y_test

    def scaling(self, X_train, X_test):
        """
        RobustScaler recommandé pour la fraude bancaire :
        robuste aux outliers (montants de transactions extrêmes).
        """
        scaler = RobustScaler()
        X_train_scaled = pd.DataFrame(
            scaler.fit_transform(X_train), columns=X_train.columns
        )
        X_test_scaled = pd.DataFrame(
            scaler.transform(X_test), columns=X_test.columns
        )
        print("Scaling (RobustScaler) effectué.")
        return X_train_scaled, X_test_scaled, scaler

    # ------------------------------------------------------------------ #
    #  APPROCHE 1 — SMOTE / resampling                                    #
    # ------------------------------------------------------------------ #

    def equilibration_smote(self, X_train, y_train, strategy="smotetomek"):
        """
        Rééquilibre les classes en créant des exemples synthétiques.
        Modifie les données → génère un nouveau X_train et y_train.

        strategy : 'smote' | 'adasyn' | 'smotetomek' | 'undersample'
        - smote       : oversample la minorité synthétiquement
        - adasyn      : smote adaptatif, focus sur les zones difficiles
        - smotetomek  : smote + nettoyage des frontières (recommandé fraude)
        - undersample : réduit la majorité (rapide mais perte d'info)
        """
        strategies = {
            "smote":       SMOTE(random_state=42),
            "adasyn":      ADASYN(random_state=42),
            "smotetomek":  SMOTETomek(random_state=42),
            "undersample": RandomUnderSampler(random_state=42),
        }
        if strategy not in strategies:
            raise ValueError(f"Strategy inconnue. Choisir parmi : {list(strategies.keys())}")

        sampler = strategies[strategy]
        X_res, y_res = sampler.fit_resample(X_train, y_train)
        X_res = pd.DataFrame(X_res, columns=X_train.columns)

        print(f"[{strategy.upper()}] Avant : {np.bincount(y_train)} → Après : {np.bincount(y_res)}")
        return X_res, y_res

    # ------------------------------------------------------------------ #
    #  APPROCHE 2 — Class weights (pas de resampling)                     #
    # ------------------------------------------------------------------ #

    def compute_class_weights(self, y_train):
        """
        Ne modifie pas les données.
        Retourne un dict {0: w0, 1: w1} à passer au paramètre
        class_weight des modèles sklearn (LR, RF, SVM).

        Utilisation :
            RandomForestClassifier(class_weight=class_weights)
            LogisticRegression(class_weight=class_weights)
            SVC(class_weight=class_weights)
        """
        from sklearn.utils.class_weight import compute_class_weight
        classes = np.unique(y_train)
        weights = compute_class_weight("balanced", classes=classes, y=y_train)
        class_weight_dict = dict(zip(classes, weights))
        print(f"Class weights calculés : {class_weight_dict}")
        return class_weight_dict

    # ------------------------------------------------------------------ #
    #  APPROCHE 3 — Anomaly score comme feature (Isolation Forest)        #
    # ------------------------------------------------------------------ #

    def add_anomaly_score(self, X_train, y_train, X_test, contamination=0.01):
        """
        Ne modifie pas l'équilibre des classes.
        Entraîne un Isolation Forest sur les transactions normales
        et ajoute une colonne 'anomaly_score' à X_train et X_test.

        Score : plus négatif = plus anormal/suspect.
        Utilisé comme feature supplémentaire pour LR, RF, SVM.
        """
        X_normal = X_train[y_train == 0]

        iso = IsolationForest(
            n_estimators=100,
            contamination=contamination,    # ~ taux de fraude attendu
            random_state=42,
            n_jobs=-1
        )
        iso.fit(X_normal)

        # Ajout du score sur train et test
        X_train_enriched = X_train.copy()
        X_test_enriched  = X_test.copy()

        X_train_enriched["anomaly_score"] = iso.decision_function(X_train)
        X_test_enriched["anomaly_score"]  = iso.decision_function(X_test)

        print(f"Anomaly score ajouté comme feature.")
        print(f"  Train → Min: {X_train_enriched['anomaly_score'].min():.3f} | Max: {X_train_enriched['anomaly_score'].max():.3f}")
        print(f"  Test  → Min: {X_test_enriched['anomaly_score'].min():.3f}  | Max: {X_test_enriched['anomaly_score'].max():.3f}")

        return X_train_enriched, X_test_enriched, iso

    def sauvegarde_csv(self, data, file_path):
        try:
            if not isinstance(data, (pd.DataFrame, pd.Series)):
                data = pd.DataFrame(data)
            data.to_csv(file_path, index=False)
            print(f"Sauvegardé → {file_path}")
        except Exception as e:
            print(f"Erreur sauvegarde : {e}")


# ------------------------------------------------------------------ #
#  PIPELINE PRINCIPAL                                                 #
# ------------------------------------------------------------------ #

if __name__ == "__main__":

    pp = DataPreprocessing()

    # 1 — Chargement
    data = pp.load_data("data/raw/creditcard.csv")

    if data is not None:

        # 2 — Nettoyage
        data = pp.remove_duplicates(data)
        data = pp.delete_features(data, ["V22", "V28", "V23", "V25","V15", "V26", "V13", "Time"])  # exemple de features à supprimer

        # 3 — Séparation X / y
        X, y = pp.separation(data, target_col="Class")

        # 4 — Split train / test (stratifié)
        X_train, X_test, y_train, y_test = pp.split_data(X, y)

        # 5 — Scaling (RobustScaler)
        # fit sur train uniquement, transform sur train et test
        X_train_scaled, X_test_scaled, scaler = pp.scaling(X_train, X_test)

        # ---------------------------------------------------------- #
        #  APPROCHE 1 — SMOTE                                         #
        #  Génère de nouvelles lignes → y_train_smote différent       #
        # ---------------------------------------------------------- #
        X_train_smote, y_train_smote = pp.equilibration_smote(
            X_train_scaled, y_train, strategy="smotetomek"
        )

        # ---------------------------------------------------------- #
        #  APPROCHE 2 — Class weights                                 #
        #  Aucune modification des données                            #
        #  À passer directement aux modèles LR, RF, SVM              #
        # ---------------------------------------------------------- #
        class_weights = pp.compute_class_weights(y_train)

        # ---------------------------------------------------------- #
        #  APPROCHE 3 — Anomaly score                                 #
        #  Ajoute une colonne anomaly_score sur train et test         #
        #  Données toujours déséquilibrées, juste enrichies           #
        # ---------------------------------------------------------- #
        X_train_anomaly, X_test_anomaly, iso_model = pp.add_anomaly_score(
            X_train_scaled, y_train, X_test_scaled, contamination=0.01
        )

        # ---------------------------------------------------------- #
        #  SAUVEGARDE                                                 #
        # ---------------------------------------------------------- #

        # Version 1 — scaled uniquement (pour IF et LOF)
        pp.sauvegarde_csv(X_train_scaled,  "data/processed/X_train_scaled.csv")
        pp.sauvegarde_csv(y_train,         "data/processed/y_train.csv")

        # Version 2 — SMOTE (pour LR, RF, SVM rééquilibrés)
        pp.sauvegarde_csv(X_train_smote,   "data/processed/X_train_smote.csv")
        pp.sauvegarde_csv(y_train_smote,   "data/processed/y_train_smote.csv")

        # Version 3 — anomaly score (pour LR, RF, SVM enrichis)
        pp.sauvegarde_csv(X_train_anomaly, "data/processed/X_train_anomaly.csv")
        pp.sauvegarde_csv(X_test_anomaly,  "data/processed/X_test_anomaly.csv")
        # y_train reste le même que la version 1 pour cette approche

        # Test set — une seule version, jamais modifié
        pp.sauvegarde_csv(X_test_scaled,   "data/processed/X_test_scaled.csv")
        pp.sauvegarde_csv(y_test,          "data/processed/y_test.csv")

        print("\n=== Récapitulatif des datasets générés ===")
        print(f"X_train_scaled  : {X_train_scaled.shape}  | y_train       : {y_train.shape}")
        print(f"X_train_smote   : {X_train_smote.shape}   | y_train_smote : {y_train_smote.shape}")
        print(f"X_train_anomaly : {X_train_anomaly.shape} | y_train       : {y_train.shape}")
        print(f"X_test_scaled   : {X_test_scaled.shape}   | y_test        : {y_test.shape}")
        print(f"\nClass weights (à passer aux modèles) : {class_weights}")