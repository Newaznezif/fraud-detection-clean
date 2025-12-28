"""
Business Recommendations Module
Task 3 - Generate actionable business recommendations
"""

import pandas as pd
import numpy as np
import os

RECOMMENDATIONS_OUTPUT_DIR = "reports/recommendations"
os.makedirs(RECOMMENDATIONS_OUTPUT_DIR, exist_ok=True)

def generate_recommendations(feature_importance_df, builtin_top_features=None, 
                           min_recommendations=3):
    """
    Generate actionable business recommendations from feature importance.
    """
    recommendations = []
    
    # Feature-specific recommendations
    for _, row in feature_importance_df.iterrows():
        feature = row['feature']
        importance = row['importance']
        
        if 'hour' in feature.lower() or 'time' in feature.lower():
            rec = {
                'type': 'temporal_analysis',
                'title': f'Time Pattern Analysis for {feature}',
                'description': f'{feature} has high importance ({abs(importance):.4f}). Time-based fraud patterns detected.',
                'action': 'Implement time-based monitoring and increase scrutiny during high-risk hours.',
                'impact': 'high',
                'priority': 'P1'
            }
            recommendations.append(rec)
        
        elif 'amount' in feature.lower() or 'value' in feature.lower():
            rec = {
                'type': 'transaction_monitoring',
                'title': f'Transaction Amount Monitoring',
                'description': f'{feature} is significant for fraud detection.',
                'action': 'Set transaction amount thresholds and require additional verification for high-value transactions.',
                'impact': 'high',
                'priority': 'P1'
            }
            recommendations.append(rec)
        
        elif 'age' in feature.lower():
            rec = {
                'type': 'demographic_analysis',
                'title': f'Age-Based Risk Assessment',
                'description': f'{feature} shows age-related fraud patterns.',
                'action': 'Analyze age distribution in fraud cases and implement age-based risk scoring.',
                'impact': 'medium',
                'priority': 'P2'
            }
            recommendations.append(rec)
    
    # General recommendations
    general_recs = [
        {
            'type': 'model_improvement',
            'title': 'Improve Model Detection Rate',
            'description': 'Current fraud detection rate can be improved.',
            'action': 'Retrain model with additional features and optimize threshold.',
            'impact': 'high',
            'priority': 'P1'
        },
        {
            'type': 'multi_layer',
            'title': 'Multi-Layer Fraud Detection',
            'description': 'Single model may miss complex fraud patterns.',
            'action': 'Combine ML model with rule-based system and manual review.',
            'impact': 'high',
            'priority': 'P1'
        },
        {
            'type': 'continuous_monitoring',
            'title': 'Continuous Model Monitoring',
            'description': 'Fraud patterns evolve over time.',
            'action': 'Implement model performance tracking and monthly retraining.',
            'impact': 'medium',
            'priority': 'P2'
        }
    ]
    
    recommendations.extend(general_recs)
    return recommendations[:min_recommendations + 3]

def print_recommendations(recommendations):
    """
    Print recommendations in formatted way.
    """
    print("\n" + "="*60)
    print("BUSINESS RECOMMENDATIONS")
    print("="*60)
    
    for i, rec in enumerate(recommendations, 1):
        print(f"\n{i}. {rec['title']}")
        print(f"   Type: {rec['type'].replace('_', ' ').title()}")
        print(f"   Priority: {rec['priority']}")
        print(f"   Description: {rec['description']}")
        print(f"   Action: {rec['action']}")
        print(f"   Impact: {rec['impact'].title()}")
