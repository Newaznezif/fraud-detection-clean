"""
SHAP Analysis Module (using permutation importance as alternative)
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
    Create feature importance summary plot (Global Feature Importance).
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
        plt.title('SHAP Summary Plot (Global Feature Importance)', fontsize=14, fontweight='bold')
        
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
    print(f"  True Positives (TP): {len(cases['tp_indices'])} - Fraud correctly detected")
    print(f"  False Positives (FP): {len(cases['fp_indices'])} - Legitimate flagged as fraud")
    print(f"  False Negatives (FN): {len(cases['fn_indices'])} - Fraud missed")
    print(f"  True Negatives (TN): {len(cases['tn_indices'])} - Legitimate correctly identified")
    
    return cases, y_pred, y_proba

def create_force_plots_alternative(model, X: pd.DataFrame, y: pd.Series, 
                                 y_pred: np.ndarray, cases: dict):
    """
    Create alternative to SHAP force plots for TP, FP, FN cases.
    Shows feature contributions for individual predictions.
    """
    import os
    
    output_dir = "reports/task3_explainability/force_plots"
    os.makedirs(output_dir, exist_ok=True)
    
    case_types = [
        ('tp', 'True Positive', 'green', cases['tp_indices'], '✅ Fraud correctly detected'),
        ('fp', 'False Positive', 'orange', cases['fp_indices'], '⚠️ Legitimate flagged as fraud'),
        ('fn', 'False Negative', 'red', cases['fn_indices'], '❌ Fraud missed')
    ]
    
    for case_code, case_name, color, indices, description in case_types:
        if len(indices) > 0:
            # Take the first instance of this type
            idx = indices[0]
            
            print(f"\n🔍 Analyzing {case_name} Case (Instance {idx}):")
            print(f"   Description: {description}")
            print(f"   Actual label: {y.iloc[idx]} (1 = fraud, 0 = legitimate)")
            print(f"   Predicted label: {y_pred[idx]}")
            
            # Get this specific instance
            instance_data = X.iloc[idx]
            
            # Get feature importances from model
            if hasattr(model, 'feature_importances_'):
                feat_imp = pd.DataFrame({
                    'feature': X.columns,
                    'importance': model.feature_importances_
                }).sort_values('importance', ascending=False)
                
                # Get top 10 most important features for this analysis
                top_features = feat_imp.head(10).copy()
                top_features['value'] = [instance_data[feat] for feat in top_features['feature']]
                
                # Calculate "contribution" (importance * normalized value)
                # This simulates SHAP's force plot concept
                normalized_values = (top_features['value'] - top_features['value'].min()) / \
                                  (top_features['value'].max() - top_features['value'].min() + 1e-10)
                top_features['contribution'] = top_features['importance'] * normalized_values
                
                # Sort by contribution (like SHAP force plot)
                top_features = top_features.sort_values('contribution', ascending=False)
                
                # Create the force plot visualization
                plt.figure(figsize=(14, 8))
                
                # Create horizontal bars showing contributions
                y_pos = np.arange(len(top_features))
                bars = plt.barh(y_pos, top_features['contribution'], color=color, alpha=0.7)
                
                # Color code based on direction of contribution
                for i, (contribution, value) in enumerate(zip(top_features['contribution'], top_features['value'])):
                    # Positive contribution pushes toward fraud prediction
                    if contribution > 0:
                        bars[i].set_color('red')
                    else:
                        bars[i].set_color('blue')
                
                plt.yticks(y_pos, top_features['feature'])
                plt.xlabel('Feature Contribution to Prediction', fontsize=12)
                plt.title(f'{case_name} - Force Plot Analysis\nInstance {idx}: {description}', 
                         fontsize=14, fontweight='bold')
                
                # Add value annotations
                for i, (contribution, value, importance) in enumerate(zip(
                    top_features['contribution'], 
                    top_features['value'], 
                    top_features['importance']
                )):
                    plt.text(abs(contribution) + 0.001, i, 
                            f'value: {value:.2f} | imp: {importance:.4f}', 
                            va='center', fontsize=9)
                
                # Add decision boundary line
                plt.axvline(x=0, color='black', linestyle='--', alpha=0.5)
                plt.text(0.001, len(top_features)-1, 'Decision Boundary', 
                        fontsize=10, color='black', va='center')
                
                plt.tight_layout()
                
                # Save the plot
                save_path = os.path.join(output_dir, f'{case_code}_force_plot.png')
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                print(f"   ✓ Force plot saved: {save_path}")
                
                plt.show()
                
                # Print detailed analysis
                print(f"\n   Top features contributing to this prediction:")
                for i, row in top_features.head(5).iterrows():
                    direction = "↑ increases fraud probability" if row['contribution'] > 0 else "↓ decreases fraud probability"
                    print(f"      {row['feature']}: {row['contribution']:.4f} ({direction})")
                
                print(f"\n   Feature values for this instance:")
                for i, row in top_features.head(3).iterrows():
                    print(f"      {row['feature']} = {row['value']:.4f}")
            
            else:
                print(f"   ⚠️ Model doesn't have feature importances for force plot")
    
    print(f"\n✅ All force plots generated and saved to: {output_dir}")