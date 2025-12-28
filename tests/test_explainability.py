"""
Unit tests for Task 3
"""

import unittest
import pandas as pd
import numpy as np
import sys
import os

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, project_root)

from src.shap_analysis import top_shap_features
from src.recommendations import generate_recommendations

class TestExplainability(unittest.TestCase):
    
    def setUp(self):
        # Create test data
        np.random.seed(42)
        self.X = pd.DataFrame({
            'feature1': np.random.randn(100),
            'amount': np.random.exponential(100, 100),
            'hour': np.random.randint(0, 24, 100)
        })
        self.shap_values = np.random.randn(100, 3)
    
    def test_top_shap_features(self):
        result = top_shap_features(self.shap_values, self.X, top_n=2)
        self.assertEqual(len(result), 2)
        self.assertIn('feature', result.columns)
        self.assertIn('mean_abs_shap', result.columns)
    
    def test_generate_recommendations(self):
        shap_top = pd.DataFrame({
            'feature': ['amount', 'hour'],
            'mean_abs_shap': [0.5, 0.3]
        })
        
        recs = generate_recommendations(shap_top, min_recommendations=2)
        self.assertGreaterEqual(len(recs), 2)
        self.assertIn('title', recs[0])
        self.assertIn('action', recs[0])

if __name__ == '__main__':
    unittest.main()