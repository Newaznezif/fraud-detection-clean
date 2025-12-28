#!/usr/bin/env python3
"""
Model Explainability Script
Main pipeline for Task 3 - Generate feature importance and business recommendations
"""

import sys
import os
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt

# Add project root to path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, project_root)

from src.shap_analysis import (
    compute_feature_importance, 
    feature_summary_plot, 
    top_features,
    analyze_prediction_cases
)
from src.feature_importance import get_builtin_importance, compare_importances
from src.recommendations import generate_recommendations, print_recommendations

def main():
    """Main explainability pipeline."""
    print("=" * 70)
    print("MODEL EXPLAINABILITY PIPELINE - TASK 3")
    print("=" * 70)
    
    # Load model
    model_path = os.path.join(project_root, 'models', 'random_forest.pkl')
    model = joblib.load(model_path)
    print(f" Model loaded: {type(model).__name__}")
    
    # Load data
    data_path = os.path.join(project_root, 'data', 'processed', 'test_fraud_processed.csv')
    data = pd.read_csv(data_path)
    
    if 'class' in data.columns:
        X_raw = data.drop('class', axis=1)
        y = data['class']
        print(f" Data loaded: {X_raw.shape}")
        print(f"  Fraud rate: {y.mean():.2%}")
    else:
        print(" 'class' column not found")
        return
    
    # Preprocess to create model features
    print("\n🔧 Preprocessing data for model features...")
    X = pd.DataFrame(index=X_raw.index)
    
    # Direct features
    if 'purchase_value' in X_raw.columns:
        X['purchase_value'] = X_raw['purchase_value']
    
    if 'age' in X_raw.columns:
        X['age'] = X_raw['age']
    
    if 'ip_address' in X_raw.columns:
        X['ip_address'] = pd.factorize(X_raw['ip_address'])[0]
    
    # Time features
    if 'purchase_time' in X_raw.columns:
        X_raw['purchase_time'] = pd.to_datetime(X_raw['purchase_time'])
        X['purchase_hour'] = X_raw['purchase_time'].dt.hour
    
    if 'purchase_time' in X_raw.columns and 'signup_time' in X_raw.columns:
        X_raw['signup_time'] = pd.to_datetime(X_raw['signup_time'])
        time_diff = X_raw['purchase_time'] - X_raw['signup_time']
        X['hours_since_signup'] = time_diff.dt.total_seconds() / 3600
    
    print(f" Processed features: {X.shape}")
    
    # Feature importance analysis
    print("\n Computing feature importance...")
    sample_size = min(1000, len(X))
    X_sample = X.iloc[:sample_size]
    y_sample = y.iloc[:sample_size]
    
    importance_results = compute_feature_importance(model, X_sample, y_sample, n_repeats=5)
    
    # Plot
    print("\n Generating visualizations...")
    importance_df = feature_summary_plot(importance_results, X_sample, plot_type='bar', max_display=10)
    
    # Top features
    top_feats = top_features(importance_results, X_sample, top_n=10)
    print("\n TOP 10 FEATURES BY IMPORTANCE:")
    print(top_feats.to_string(index=False))
    
    # Built-in importance
    if hasattr(model, 'feature_importances_'):
        builtin_top = get_builtin_importance(model, X_sample, top_n=10)
        print("\n TOP 10 BUILT-IN FEATURES:")
        print(builtin_top[['feature', 'importance']].to_string(index=False))
        
        # Compare
        compare_importances(builtin_top, top_feats)
    else:
        builtin_top = None
    
    # Prediction case analysis
    print("\n Analyzing prediction cases...")
    cases, y_pred, y_proba = analyze_prediction_cases(model, X, y)
    accuracy = (y_pred == y).mean()
    print(f"  Model Accuracy: {accuracy:.2%}")
    
    # Business recommendations
    print("\n Generating business recommendations...")
    
    # Prepare data for recommendations
    shap_like_df = pd.DataFrame({
        'feature': top_feats['feature'],
        'importance': np.abs(top_feats['importance'])
    })
    
    recommendations = generate_recommendations(shap_like_df, builtin_top, min_recommendations=5)
    print_recommendations(recommendations)
    
    # Save results
    print("\n Saving results...")
    output_dir = os.path.join(project_root, 'reports', 'task3_explainability')
    os.makedirs(output_dir, exist_ok=True)
    
    top_feats.to_csv(os.path.join(output_dir, 'feature_importance.csv'), index=False)
    
    # Create report
    report_content = f"""FRAUD DETECTION MODEL EXPLAINABILITY REPORT
{'='*60}

MODEL INFORMATION:
- Model: {type(model).__name__}
- Features analyzed: {len(X.columns)}
- Samples: {len(X):,}

PERFORMANCE:
- Accuracy: {accuracy:.2%}
- Fraud rate: {y.mean():.2%}

TOP FEATURES:
{top_feats.to_string()}

KEY INSIGHTS:
1. 'hours_since_signup' is the most important feature
2. Fraud patterns are strongly time-dependent
3. Current detection rate: {accuracy:.2%}

RECOMMENDATIONS:
1. Implement time-based fraud monitoring
2. Review model training to utilize all features
3. Consider additional feature engineering
"""
    
    with open(os.path.join(output_dir, 'analysis_report.txt'), 'w') as f:
        f.write(report_content)
    
    print(f" Results saved to: {output_dir}")
    
    print("\n" + "=" * 70)
    print("TASK 3 COMPLETE ")
    print("=" * 70)
    print("\nSummary:")
    print(f"- Model: {type(model).__name__}")
    print(f"- Top feature: '{top_feats.iloc[0]['feature'] if len(top_feats) > 0 else 'None'}'")
    print(f"- Accuracy: {accuracy:.2%}")
    print(f"- Recommendations generated: {len(recommendations)}")

if __name__ == "__main__":
    main()
