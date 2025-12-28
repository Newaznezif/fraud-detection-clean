"""
SHAP Analysis Module (using permutation importance)
Task 3 - Model Explainability
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from sklearn.inspection import permutation_importance
import warnings
warnings.filterwarnings('ignore')

SHAP_OUTPUT_DIR = "reports/shap_plots"
os.makedirs(SHAP_OUTPUT_DIR, exist_ok=True)

def compute_feature_importance(model, X: pd.DataFrame, y: pd.Series, 
                             n_repeats: int = 10, random_state: int = 42):
    """
    Compute feature importance using permutation importance.
    """
    print("Computing permutation importance...")
    
    perm_importance = permutation_importance(
        model, X, y,
        n_repeats=n_repeats,
        random_state=random_state,
        n_jobs=-1
    )
    
    return {
        'importances_mean': perm_importance.importances_mean,
        'importances_std': perm_importance.importances_std,
        'importances': perm_importance.importances
    }

def feature_summary_plot(importance_results, X: pd.DataFrame, 
                        plot_type: str = "bar", max_display: int = 20,
                        save_path: str = None):
    """
    Create feature importance summary plot.
    """
    feature_names = X.columns.tolist()
    importances_mean = importance_results['importances_mean']
    importances_std = importance_results['importances_std']
    
    importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance_mean': importances_mean,
        'importance_std': importances_std
    }).sort_values('importance_mean', ascending=False).head(max_display)
    
    plt.figure(figsize=(12, 8))
    
    if plot_type == "bar":
        bars = plt.barh(range(len(importance_df)), 
                       importance_df['importance_mean'][::-1])
        plt.yticks(range(len(importance_df)), importance_df['feature'][::-1])
        plt.xlabel('Permutation Importance')
        plt.title('Feature Importance (Permutation)', fontsize=14, fontweight='bold')
        
        for i, (idx, row) in enumerate(importance_df[::-1].iterrows()):
            plt.errorbar(row['importance_mean'], i, 
                        xerr=row['importance_std'], 
                        color='black', capsize=5)
    
    elif plot_type == "dot":
        y_pos = np.arange(len(importance_df))
        plt.scatter(importance_df['importance_mean'], y_pos[::-1], 
                   s=100, alpha=0.7)
        plt.yticks(y_pos, importance_df['feature'][::-1])
        plt.xlabel('Importance')
        plt.title('Feature Importance Distribution', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved: {save_path}")
    
    plt.show()
    return importance_df

def top_features(importance_results, X: pd.DataFrame, top_n: int = 10):
    """
    Extract top features by mean absolute importance.
    """
    feature_names = X.columns.tolist()
    importances_mean = importance_results['importances_mean']
    
    features_df = pd.DataFrame({
        'feature': feature_names,
        'importance': importances_mean,
        'importance_abs': np.abs(importances_mean)
    })
    
    top_features_df = features_df.sort_values('importance_abs', 
                                            ascending=False).head(top_n)
    top_features_df = top_features_df[['feature', 'importance']]
    top_features_df = top_features_df.reset_index(drop=True)
    top_features_df.index = top_features_df.index + 1
    
    return top_features_df

def analyze_prediction_cases(model, X: pd.DataFrame, y_true: pd.Series, 
                           threshold: float = 0.5):
    """
    Analyze prediction cases (TP, FP, FN).
    """
    if hasattr(model, 'predict_proba'):
        y_proba = model.predict_proba(X)[:, 1]
        y_pred = (y_proba >= threshold).astype(int)
    else:
        y_pred = model.predict(X)
        y_proba = y_pred
    
    from sklearn.metrics import confusion_matrix
    
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()
    
    cases = {
        'tp_indices': np.where((y_true == 1) & (y_pred == 1))[0],
        'fp_indices': np.where((y_true == 0) & (y_pred == 1))[0],
        'fn_indices': np.where((y_true == 1) & (y_pred == 0))[0],
        'tn_indices': np.where((y_true == 0) & (y_pred == 0))[0]
    }
    
    print("Prediction Case Analysis:")
    print(f"  True Positives (TP): {len(cases['tp_indices'])}")
    print(f"  False Positives (FP): {len(cases['fp_indices'])}")
    print(f"  False Negatives (FN): {len(cases['fn_indices'])}")
    print(f"  True Negatives (TN): {len(cases['tn_indices'])}")
    
    return cases, y_pred, y_proba
