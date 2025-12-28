"""
Business Recommendations Module
Task 3 - Generate actionable business recommendations
Required: Provide at least 3 actionable recommendations based on your analysis
"""

import pandas as pd
import numpy as np
import os

RECOMMENDATIONS_OUTPUT_DIR = "reports/recommendations"
os.makedirs(RECOMMENDATIONS_OUTPUT_DIR, exist_ok=True)

def generate_recommendations(feature_importance_df, builtin_top_features=None, 
                           case_analysis=None, min_recommendations=3):
    """
    Generate actionable business recommendations from feature importance.
    Required: Connect each recommendation to specific SHAP insights
    """
    recommendations = []
    
    print("=" * 60)
    print("BUSINESS RECOMMENDATIONS GENERATION")
    print("=" * 60)
    
    # 1. Recommendations based on top features
    print(f"\n📊 Analyzing top {len(feature_importance_df)} important features...")
    
    for i, (_, row) in enumerate(feature_importance_df.iterrows()):
        feature = row['feature']
        importance = abs(row['importance'])
        
        # Recommendation 1: Time-based features
        if any(time_word in feature.lower() for time_word in ['hour', 'time', 'day', 'week', 'since']):
            rec = {
                'id': f"REC-TIME-{i+1:03d}",
                'type': 'temporal_monitoring',
                'title': f'Time-Based Fraud Detection for "{feature}"',
                'description': f'Feature "{feature}" has high importance ({importance:.4f}). '
                             f'Fraud patterns show strong time dependence.',
                'action': 'Implement time-based monitoring rules. Flag transactions during high-risk time periods '
                         'identified by the model. For example, increase scrutiny for transactions occurring '
                         'at unusual hours or specific time intervals after account creation.',
                'impact': 'high',
                'effort': 'medium',
                'priority': 'P1',
                'connected_to': f'SHAP insight: "{feature}" is a top predictor of fraud',
                'expected_benefit': 'Reduce fraud by targeting time-based patterns'
            }
            recommendations.append(rec)
            print(f"  ⏰ Time-based recommendation: {feature}")
        
        # Recommendation 2: Transaction amount features
        elif any(amt_word in feature.lower() for amt_word in ['amount', 'value', 'price', 'purchase']):
            rec = {
                'id': f"REC-AMT-{i+1:03d}",
                'type': 'transaction_monitoring',
                'title': f'Transaction Amount Monitoring for "{feature}"',
                'description': f'"{feature}" significantly impacts fraud predictions ({importance:.4f}). '
                             f'Transaction values are strong fraud indicators.',
                'action': 'Set transaction amount thresholds and implement tiered verification. '
                         'Require additional authentication for transactions above specific amounts. '
                         'Monitor for unusual spending patterns and velocity.',
                'impact': 'high',
                'effort': 'low',
                'priority': 'P1',
                'connected_to': f'SHAP insight: "{feature}" value strongly influences fraud probability',
                'expected_benefit': 'Catch high-value fraud and unusual spending patterns'
            }
            recommendations.append(rec)
            print(f"  💰 Amount-based recommendation: {feature}")
        
        # Recommendation 3: User/profile features
        elif any(user_word in feature.lower() for user_word in ['age', 'user', 'profile', 'device', 'ip']):
            rec = {
                'id': f"REC-USER-{i+1:03d}",
                'type': 'user_behavior',
                'title': f'User Behavior Analysis for "{feature}"',
                'description': f'"{feature}" provides important user context ({importance:.4f}). '
                             f'User characteristics and behavior patterns affect fraud risk.',
                'action': 'Implement user profiling and behavior analysis. Monitor for anomalies in '
                         'user behavior patterns. Consider risk scoring based on user attributes '
                         'and historical behavior.',
                'impact': 'medium',
                'effort': 'high',
                'priority': 'P2',
                'connected_to': f'SHAP insight: "{feature}" contributes to user risk assessment',
                'expected_benefit': 'Identify suspicious user behavior and account takeover attempts'
            }
            recommendations.append(rec)
            print(f"  👤 User-based recommendation: {feature}")
    
    # 2. General recommendations based on analysis
    print(f"\n📈 Adding general recommendations based on overall analysis...")
    
    general_recommendations = [
        {
            'id': 'REC-GEN-001',
            'type': 'model_improvement',
            'title': 'Improve Model Detection Rate',
            'description': 'Current model shows room for improvement in fraud detection.',
            'action': '1. Retrain model with additional features\n'
                     '2. Implement ensemble of multiple models\n'
                     '3. Add real-time feedback loop\n'
                     '4. Optimize prediction threshold based on business costs',
            'impact': 'high',
            'effort': 'high',
            'priority': 'P1',
            'connected_to': 'Overall model performance analysis',
            'expected_benefit': 'Increase fraud detection rate while maintaining acceptable false positive rate'
        },
        {
            'id': 'REC-GEN-002',
            'type': 'multi_layer',
            'title': 'Implement Multi-Layer Fraud Detection',
            'description': 'No single model catches all fraud. Defense-in-depth approach is essential.',
            'action': '1. First layer: ML model (current implementation)\n'
                     '2. Second layer: Rule-based system (business rules)\n'
                     '3. Third layer: Manual review for edge cases\n'
                     '4. Fourth layer: Continuous learning from new patterns',
            'impact': 'high',
            'effort': 'medium',
            'priority': 'P1',
            'connected_to': 'Analysis of false positives and false negatives',
            'expected_benefit': 'Comprehensive fraud coverage with reduced false positives'
        },
        {
            'id': 'REC-GEN-003',
            'type': 'continuous_monitoring',
            'title': 'Continuous Model Monitoring and Retraining',
            'description': 'Fraud patterns evolve over time. Static models become less effective.',
            'action': '1. Implement model performance tracking dashboard\n'
                     '2. Set up automated retraining pipeline (monthly)\n'
                     '3. Monitor feature drift and concept drift\n'
                     '4. Regular A/B testing of new model versions',
            'impact': 'medium',
            'effort': 'medium',
            'priority': 'P2',
            'connected_to': 'Time-based pattern analysis',
            'expected_benefit': 'Maintain model effectiveness as fraud patterns change'
        }
    ]
    
    recommendations.extend(general_recommendations)
    
    # 3. Ensure minimum number of recommendations
    if len(recommendations) < min_recommendations:
        additional = [
            {
                'id': 'REC-ADD-001',
                'type': 'data_quality',
                'title': 'Improve Data Collection and Quality',
                'description': 'Better data leads to better fraud detection.',
                'action': 'Collect additional features: device fingerprint, behavioral biometrics, '
                         'network information, transaction context.',
                'impact': 'medium',
                'effort': 'high',
                'priority': 'P3',
                'connected_to': 'Feature importance analysis shows limited feature set',
                'expected_benefit': 'Enable more sophisticated fraud detection patterns'
            }
        ]
        recommendations.extend(additional[:min_recommendations - len(recommendations)])
    
    print(f"\n✅ Generated {len(recommendations)} recommendations")
    return recommendations[:10]  # Return top 10 recommendations

def print_recommendations(recommendations):
    """
    Print recommendations in formatted way.
    """
    print("\n" + "=" * 70)
    print("ACTIONABLE BUSINESS RECOMMENDATIONS")
    print("=" * 70)
    
    for i, rec in enumerate(recommendations, 1):
        print(f"\n{i}. {rec['title']} [{rec['id']}]")
        print(f"   Type: {rec['type'].replace('_', ' ').title()}")
        print(f"   Priority: {rec['priority']}")
        print(f"\n   Description:")
        print(f"   {rec['description']}")
        print(f"\n   Recommended Action:")
        print(f"   {rec['action']}")
        print(f"\n   Connected to: {rec['connected_to']}")
        print(f"   Expected Benefit: {rec['expected_benefit']}")
        print(f"   Impact: {rec['impact'].title()} | Effort: {rec['effort'].title()}")
        print("-" * 70)
    
    # Summary
    print(f"\n📋 SUMMARY: {len(recommendations)} recommendations")
    
    # Count by priority
    priorities = {}
    for rec in recommendations:
        priorities[rec['priority']] = priorities.get(rec['priority'], 0) + 1
    
    print("\nPriority Distribution:")
    for priority, count in sorted(priorities.items()):
        print(f"  {priority}: {count} recommendations")
    
    # Save to file
    rec_df = pd.DataFrame(recommendations)
    save_path = os.path.join(RECOMMENDATIONS_OUTPUT_DIR, "business_recommendations.csv")
    rec_df.to_csv(save_path, index=False)
    print(f"\n💾 Recommendations saved to: {save_path}")