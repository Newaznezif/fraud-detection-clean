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
    Extract built-in feature importances from ensemble model.
    """
    if hasattr(model, 'feature_importances_'):
        importance = model.feature_importances_
        df = pd.DataFrame({
            "feature": X.columns,
            "importance": importance
        }).sort_values("importance", ascending=False)
        
        print(f"Extracted built-in importance for {len(df)} features")
        return df.head(top_n)
    else:
        print("⚠️ Model doesn't have built-in feature importances")
        return pd.DataFrame(columns=["feature", "importance"])

def compare_importances(builtin_df, perm_df, save_path=None):
    """
    Compare model-intrinsic feature importance with permutation importance.
    Required: Compare SHAP importance with built-in feature importance
    """
    if builtin_df.empty or perm_df.empty:
        print("⚠️ Cannot compare: One or both DataFrames are empty")
        return None
    
    # Rename columns for clarity
    builtin_renamed = builtin_df.rename(columns={'importance': 'importance_builtin'})
    perm_renamed = perm_df.rename(columns={'importance': 'permutation_importance'})
    
    # Merge the two importance DataFrames
    merged = pd.merge(
        builtin_renamed, 
        perm_renamed,
        on="feature", 
        how="outer"
    ).fillna(0)

    # Sort for better visualization
    merged = merged.sort_values("importance_builtin", ascending=True)

    plt.figure(figsize=(12, 8))
    
    # Plot both importances
    y_pos = np.arange(len(merged))
    width = 0.35
    
    plt.barh(y_pos - width/2, merged["importance_builtin"], 
             height=width, alpha=0.7, label="Built-in Importance", color='blue')
    plt.barh(y_pos + width/2, merged["permutation_importance"], 
             height=width, alpha=0.7, label="Permutation Importance", color='red')
    
    plt.yticks(y_pos, merged["feature"])
    plt.xlabel("Importance Score", fontsize=12)
    plt.title("Feature Importance Comparison: Built-in vs Permutation", 
             fontsize=14, fontweight='bold')
    plt.legend()
    
    # Add correlation value
    correlation = merged["importance_builtin"].corr(merged["permutation_importance"])
    plt.text(0.02, 0.98, f'Correlation: {correlation:.3f}', 
             transform=plt.gca().transAxes, fontsize=11,
             verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()

    if save_path:
        if save_path == "auto":
            save_path = os.path.join(FEATURE_OUTPUT_DIR, "importance_comparison.png")
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"✓ Importance comparison saved → {save_path}")
    
    plt.show()
    
    # Print comparison analysis
    print("\n📊 IMPORTANCE COMPARISON ANALYSIS:")
    print("=" * 50)
    
    # Find top 5 features by each method
    top_builtin = merged.nlargest(5, 'importance_builtin')[['feature', 'importance_builtin']]
    top_perm = merged.nlargest(5, 'permutation_importance')[['feature', 'permutation_importance']]
    
    print("\nTop 5 Features by Built-in Importance:")
    for i, row in top_builtin.iterrows():
        print(f"  {row['feature']}: {row['importance_builtin']:.4f}")
    
    print("\nTop 5 Features by Permutation Importance:")
    for i, row in top_perm.iterrows():
        print(f"  {row['feature']}: {row['permutation_importance']:.4f}")
    
    # Find features that appear in both top 5 lists
    common_features = set(top_builtin['feature']).intersection(set(top_perm['feature']))
    if common_features:
        print(f"\n✅ Common important features: {', '.join(common_features)}")
        print("   These features are consistently important across both methods.")
    else:
        print("\n⚠️ No common features in top 5 - methods disagree on importance")
    
    print(f"\nCorrelation between methods: {correlation:.3f}")
    if correlation > 0.7:
        print("✅ High correlation - methods largely agree")
    elif correlation > 0.3:
        print("⚠️ Moderate correlation - some disagreement")
    else:
        print("❌ Low correlation - methods disagree significantly")
    
    return merged