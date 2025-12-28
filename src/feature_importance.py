"""
Feature Importance Module
Task 3 - Compare built-in vs permutation importance
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

FEATURE_OUTPUT_DIR = "reports/feature_importance"
os.makedirs(FEATURE_OUTPUT_DIR, exist_ok=True)

def get_builtin_importance(model, X, top_n=10):
    """
    Extract feature importances from ensemble model.
    """
    importance = model.feature_importances_
    df = pd.DataFrame({
        "feature": X.columns,
        "importance": importance
    }).sort_values("importance", ascending=False)
    return df.head(top_n)

def compare_importances(builtin_df, perm_df, save_path=None):
    """
    Compare model-intrinsic feature importance with permutation importance.
    """
    merged = pd.merge(
        builtin_df, perm_df, on="feature", how="outer", 
        suffixes=("_builtin", "_perm")
    ).fillna(0)

    merged = merged.sort_values("importance_builtin", ascending=True)

    plt.figure(figsize=(10, 6))
    plt.barh(merged["feature"], merged["importance_builtin"], 
             alpha=0.6, label="Built-in")
    plt.barh(merged["feature"], merged["importance"], 
             alpha=0.6, label="Permutation")
    plt.xlabel("Importance")
    plt.title("Feature Importance: Built-in vs Permutation")
    plt.legend()

    if save_path:
        plt.savefig(save_path, bbox_inches="tight")
        print(f"Feature importance comparison saved  {save_path}")

    plt.show()
    return merged
