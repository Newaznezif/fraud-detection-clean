"""
Individual Predictions Explanations
Task 3 - Generate explanations for TP, FP, FN cases
"""

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix
import os

INDIVIDUAL_OUTPUT_DIR = "reports/individual_explanations"
os.makedirs(INDIVIDUAL_OUTPUT_DIR, exist_ok=True)

def find_prediction_cases(model, X_test, y_test, threshold=0.5):
    """
    Identify indices of TP, FP, and FN cases.
    Required: Generate SHAP Force Plots for at least 3 individual predictions
    """
    # Get predictions
    if hasattr(model, 'predict_proba'):
        y_proba = model.predict_proba(X_test)[:, 1]
        y_pred = (y_proba >= threshold).astype(int)
    else:
        y_pred = model.predict(X_test)
        y_proba = y_pred
    
    # Get confusion matrix
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
    
    # Find indices for each case type
    cases = {
        "tp_indices": np.where((y_test == 1) & (y_pred == 1))[0],
        "fp_indices": np.where((y_test == 0) & (y_pred == 1))[0],
        "fn_indices": np.where((y_test == 1) & (y_pred == 0))[0],
        "tn_indices": np.where((y_test == 0) & (y_pred == 0))[0],
        "y_pred": y_pred,
        "y_proba": y_proba
    }
    
    # Print detailed statistics
    print("=" * 60)
    print("PREDICTION CASE ANALYSIS")
    print("=" * 60)
    
    print(f"\n📊 Confusion Matrix:")
    print(f"                   Predicted")
    print(f"                  Fraud  Legitimate")
    print(f"  Actual Fraud     {tp:6d}    {fn:6d}")
    print(f"  Actual Legitimate {fp:6d}    {tn:6d}")
    
    print(f"\n🔍 Case Details:")
    print(f"  True Positives (TP): {len(cases['tp_indices']):,} - Fraud correctly detected")
    print(f"  False Positives (FP): {len(cases['fp_indices']):,} - Legitimate flagged as fraud")
    print(f"  False Negatives (FN): {len(cases['fn_indices']):,} - Fraud missed")
    print(f"  True Negatives (TN): {len(cases['tn_indices']):,} - Legitimate correctly identified")
    
    # Calculate rates
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    print(f"\n📈 Performance Metrics:")
    print(f"  Accuracy: {(tp + tn) / len(y_test):.2%}")
    print(f"  Precision: {precision:.2%} (of predicted fraud, how many are actual fraud)")
    print(f"  Recall: {recall:.2%} (of actual fraud, how many are detected)")
    print(f"  F1-Score: {f1:.2%}")
    print(f"  False Positive Rate: {fp / (fp + tn):.2%}")
    
    # Save case indices for later use
    case_report = {
        'tp_count': len(cases['tp_indices']),
        'fp_count': len(cases['fp_indices']),
        'fn_count': len(cases['fn_indices']),
        'tn_count': len(cases['tn_indices']),
        'accuracy': (tp + tn) / len(y_test),
        'precision': precision,
        'recall': recall,
        'f1_score': f1
    }
    
    # Save to file
    report_df = pd.DataFrame([case_report])
    report_path = os.path.join(INDIVIDUAL_OUTPUT_DIR, "prediction_cases_report.csv")
    report_df.to_csv(report_path, index=False)
    print(f"\n💾 Case analysis saved to: {report_path}")
    
    return cases