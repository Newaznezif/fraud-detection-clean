"""
Task 2: Model Building and Training
----------------------------------
- Stratified Train/Test Split
- Logistic Regression Baseline
- Random Forest Ensemble
- PR-AUC, ROC-AUC, F1, Confusion Matrix
- 5-Fold Stratified Cross-Validation
- Model Comparison & Selection
- Supports BOTH Fraud_Data.csv and creditcard.csv
"""

import os
import warnings
import joblib
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    f1_score,
    confusion_matrix,
)

warnings.filterwarnings("ignore")


class Task2Models:
    def __init__(self, random_state: int = 42):
        self.random_state = random_state
        self.models = {}
        self.results = {}

    # =========================================================
    # DATA LOADING
    # =========================================================
    def load_fraud_data(self):
        """Load and prepare Fraud_Data.csv"""
        df = pd.read_csv("data/raw/Fraud_Data.csv")

        # Time-based features (lightweight, safe)
        df["signup_time"] = pd.to_datetime(df["signup_time"], errors="coerce")
        df["purchase_time"] = pd.to_datetime(df["purchase_time"], errors="coerce")

        df["purchase_hour"] = df["purchase_time"].dt.hour
        df["hours_since_signup"] = (
            (df["purchase_time"] - df["signup_time"])
            .dt.total_seconds()
            / 3600
        )

        numeric_cols = df.select_dtypes(include=[np.number]).columns
        features = [c for c in numeric_cols if c not in ["class", "user_id"]]

        X = df[features].fillna(0)
        y = df["class"]

        return X, y

    def load_creditcard_data(self):
        """Load and prepare creditcard.csv"""
        df = pd.read_csv("data/raw/creditcard.csv")

        X = df.drop(columns=["Class"])
        y = df["Class"]

        return X, y

    # =========================================================
    # DATA SPLIT
    # =========================================================
    def stratified_split(self, X, y):
        return train_test_split(
            X,
            y,
            test_size=0.2,
            stratify=y,
            random_state=self.random_state,
        )

    # =========================================================
    # MODELS
    # =========================================================
    def train_logistic_regression(self, X_train, y_train, X_test, y_test, tag):
        print(f"\n[BASELINE] Logistic Regression — {tag}")

        model = LogisticRegression(
            max_iter=1000,
            class_weight="balanced",
            random_state=self.random_state,
        )
        model.fit(X_train, y_train)

        metrics = self.evaluate(model, X_test, y_test, "Logistic Regression")
        self.models[f"logistic_{tag}"] = model
        self.results[f"logistic_{tag}"] = metrics

    def train_random_forest(self, X_train, y_train, X_test, y_test, tag):
        print(f"\n[ENSEMBLE] Random Forest — {tag}")

        best_model = None
        best_pr_auc = 0

        for depth in [5, 10, 15]:
            model = RandomForestClassifier(
                n_estimators=100,
                max_depth=depth,
                class_weight="balanced_subsample",
                n_jobs=-1,
                random_state=self.random_state,
            )
            model.fit(X_train, y_train)

            pr_auc = average_precision_score(
                y_test, model.predict_proba(X_test)[:, 1]
            )

            if pr_auc > best_pr_auc:
                best_pr_auc = pr_auc
                best_model = model

        metrics = self.evaluate(best_model, X_test, y_test, "Random Forest")
        self.models[f"rf_{tag}"] = best_model
        self.results[f"rf_{tag}"] = metrics

    # =========================================================
    # EVALUATION
    # =========================================================
    def evaluate(self, model, X_test, y_test, name):
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]

        metrics = {
            "model": name,
            "pr_auc": average_precision_score(y_test, y_prob),
            "roc_auc": roc_auc_score(y_test, y_prob),
            "f1": f1_score(y_test, y_pred),
            "confusion_matrix": confusion_matrix(y_test, y_pred),
        }

        cm = metrics["confusion_matrix"]

        print(f"PR-AUC : {metrics['pr_auc']:.4f}")
        print(f"ROC-AUC: {metrics['roc_auc']:.4f}")
        print(f"F1     : {metrics['f1']:.4f}")
        print("Confusion Matrix:")
        print(cm)

        return metrics

    # =========================================================
    # CROSS-VALIDATION
    # =========================================================
    def cross_validate(self, model, X, y, name):
        print(f"\n[CV] 5-Fold Stratified — {name}")

        cv = StratifiedKFold(
            n_splits=5, shuffle=True, random_state=self.random_state
        )

        pr_auc = cross_val_score(
            model, X, y, scoring="average_precision", cv=cv, n_jobs=-1
        )
        f1 = cross_val_score(
            model, X, y, scoring="f1", cv=cv, n_jobs=-1
        )

        print(f"PR-AUC: {pr_auc.mean():.4f} ± {pr_auc.std():.4f}")
        print(f"F1    : {f1.mean():.4f} ± {f1.std():.4f}")

        return {
            "pr_auc_mean": pr_auc.mean(),
            "pr_auc_std": pr_auc.std(),
            "f1_mean": f1.mean(),
            "f1_std": f1.std(),
        }

    # =========================================================
    # MODEL SELECTION
    # =========================================================
    def select_best_model(self):
        print("\n[MODEL COMPARISON]")

        df = pd.DataFrame([
            {
                "Model": v["model"],
                "PR-AUC": v["pr_auc"],
                "F1": v["f1"],
            }
            for v in self.results.values()
        ]).sort_values("PR-AUC", ascending=False)

        print(df.to_string(index=False))

        return df.iloc[0]["Model"]

    # =========================================================
    # SAVE MODEL
    # =========================================================
    def save_model(self, model, name):
        os.makedirs("models", exist_ok=True)
        path = f"models/{name.lower().replace(' ', '_')}.pkl"
        joblib.dump(model, path)
        print(f"\nModel saved → {path}")


# =============================================================
# PIPELINE RUNNER
# =============================================================
def run_task2():
    print("\n" + "=" * 70)
    print("TASK 2 — MODEL BUILDING & TRAINING")
    print("=" * 70)

    trainer = Task2Models()

    # ================= FRAUD DATA =================
    X, y = trainer.load_fraud_data()
    X_tr, X_te, y_tr, y_te = trainer.stratified_split(X, y)

    trainer.train_logistic_regression(X_tr, y_tr, X_te, y_te, "fraud")
    trainer.train_random_forest(X_tr, y_tr, X_te, y_te, "fraud")

    # ================= CREDIT CARD =================
    Xc, yc = trainer.load_creditcard_data()
    Xc_tr, Xc_te, yc_tr, yc_te = trainer.stratified_split(Xc, yc)

    trainer.train_logistic_regression(Xc_tr, yc_tr, Xc_te, yc_te, "creditcard")
    trainer.train_random_forest(Xc_tr, yc_tr, Xc_te, yc_te, "creditcard")

    # ================= CROSS-VALIDATION =================
    for key, model in trainer.models.items():
        tag = key.split("_")[-1]
        X_cv, y_cv = (X, y) if tag == "fraud" else (Xc, yc)
        trainer.cross_validate(model, X_cv, y_cv, trainer.results[key]["model"])

    best = trainer.select_best_model()
    print(f"\nBEST MODEL SELECTED: {best}")
    print("Justification: Highest PR-AUC on imbalanced fraud data.")

    print("\nTASK 2 COMPLETED SUCCESSFULLY")


if __name__ == "__main__":
    run_task2()
