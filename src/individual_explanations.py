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

def find_prediction_cases(model, X_test, y_test):
    """
    Identify indices of TP, FP, and FN cases.
    """
    y_pred = model.predict(X_test)
    
    # Get confusion matrix
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
    
    # Find indices for each case
    cases = {
        "tp_indices": np.where((y_test == 1) & (y_pred == 1))[0],
        "fp_indices": np.where((y_test == 0) & (y_pred == 1))[0],
        "fn_indices": np.where((y_test == 1) & (y_pred == 0))[0],
        "tn_indices": np.where((y_test == 0) & (y_pred == 0))[0]
    }
    
    # Print statistics
    print("Prediction Case Statistics:")
    print(f"  True Positives (TP): {len(cases['tp_indices'])}")
    print(f"  False Positives (FP): {len(cases['fp_indices'])}")
    print(f"  False Negatives (FN): {len(cases['fn_indices'])}")
    print(f"  True Negatives (TN): {len(cases['tn_indices'])}")
    
    return cases
