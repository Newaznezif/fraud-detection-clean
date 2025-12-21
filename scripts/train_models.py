#!/usr/bin/env python
"""
Model training script for fraud detection
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_predict
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                           f1_score, roc_auc_score, average_precision_score,
                           confusion_matrix, classification_report, roc_curve, 
                           precision_recall_curve)
import xgboost as xgb
import lightgbm as lgb
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

class FraudModelTrainer:
    """Train and evaluate fraud detection models"""
    
    def __init__(self, random_state=42):
        self.random_state = random_state
        self.models = {}
        self.results = {}
        self.scaler = StandardScaler()
        self.best_model = None
        self.best_model_name = None
        
    def load_data(self, data_path="../data/processed/fraud_data_processed.csv"):
        """Load processed data"""
        print("Loading processed data...")
        df = pd.read_csv(data_path)
        print(f"Data shape: {df.shape}")
        
        # Separate features and target
        X = df.drop(columns=['class', 'user_id', 'device_id', 'ip_address', 
                            'signup_time', 'purchase_time', 'country'])
        
        # Handle remaining non-numeric columns
        for col in X.select_dtypes(include=['object']).columns:
            X[col] = pd.factorize(X[col])[0]
        
        y = df['class']
        
        print(f"Features shape: {X.shape}")
        print(f"Target distribution: {y.value_counts().to_dict()}")
        
        return X, y, df
    
    def prepare_data(self, X, y, test_size=0.3):
        """Prepare train-test split"""
        print(f"\nSplitting data (test_size={test_size})...")
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, stratify=y, random_state=self.random_state
        )
        
        print(f"Train shape: {X_train.shape}, Fraud rate: {y_train.mean():.4f}")
        print(f"Test shape: {X_test.shape}, Fraud rate: {y_test.mean():.4f}")
        
        return X_train, X_test, y_train, y_test
    
    def handle_imbalance(self, X_train, y_train, method='smote'):
        """Handle class imbalance in training data"""
        print(f"\nHandling class imbalance using {method}...")
        print(f"Before: Class 0: {sum(y_train==0):,}, Class 1: {sum(y_train==1):,}")
        
        if method == 'smote':
            smote = SMOTE(random_state=self.random_state, sampling_strategy=0.3)
            X_res, y_res = smote.fit_resample(X_train, y_train)
        elif method == 'undersample':
            rus = RandomUnderSampler(random_state=self.random_state, sampling_strategy=0.5)
            X_res, y_res = rus.fit_resample(X_train, y_train)
        elif method == 'combined':
            # First oversample, then undersample
            smote = SMOTE(random_state=self.random_state, sampling_strategy=0.5)
            X_over, y_over = smote.fit_resample(X_train, y_train)
            rus = RandomUnderSampler(random_state=self.random_state, sampling_strategy=0.8)
            X_res, y_res = rus.fit_resample(X_over, y_over)
        else:
            X_res, y_res = X_train, y_train
        
        print(f"After: Class 0: {sum(y_res==0):,}, Class 1: {sum(y_res==1):,}")
        
        return X_res, y_res
    
    def train_logistic_regression(self, X_train, y_train):
        """Train logistic regression model"""
        print("\n" + "="*50)
        print("Training Logistic Regression")
        print("="*50)
        
        model = LogisticRegression(
            random_state=self.random_state,
            class_weight='balanced',
            max_iter=1000,
            C=0.1,
            solver='liblinear'
        )
        
        model.fit(X_train, y_train)
        self.models['logistic_regression'] = model
        
        return model
    
    def train_random_forest(self, X_train, y_train):
        """Train random forest model"""
        print("\n" + "="*50)
        print("Training Random Forest")
        print("="*50)
        
        model = RandomForestClassifier(
            n_estimators=200,
            max_depth=15,
            min_samples_split=5,
            min_samples_leaf=2,
            class_weight='balanced',
            random_state=self.random_state,
            n_jobs=-1,
            max_features='sqrt'
        )
        
        model.fit(X_train, y_train)
        self.models['random_forest'] = model
        
        return model
    
    def train_xgboost(self, X_train, y_train):
        """Train XGBoost model"""
        print("\n" + "="*50)
        print("Training XGBoost")
        print("="*50)
        
        # Calculate scale_pos_weight for imbalance
        scale_pos_weight = sum(y_train == 0) / sum(y_train == 1)
        
        model = xgb.XGBClassifier(
            n_estimators=150,
            max_depth=8,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            scale_pos_weight=scale_pos_weight,
            random_state=self.random_state,
            use_label_encoder=False,
            eval_metric='logloss',
            n_jobs=-1
        )
        
        model.fit(X_train, y_train)
        self.models['xgboost'] = model
        
        return model
    
    def train_lightgbm(self, X_train, y_train):
        """Train LightGBM model"""
        print("\n" + "="*50)
        print("Training LightGBM")
        print("="*50)
        
        model = lgb.LGBMClassifier(
            n_estimators=150,
            max_depth=10,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            is_unbalance=True,
            random_state=self.random_state,
            n_jobs=-1,
            boosting_type='gbdt'
        )
        
        model.fit(X_train, y_train)
        self.models['lightgbm'] = model
        
        return model
    
    def evaluate_model(self, model, X_test, y_test, model_name):
        """Evaluate model performance"""
        print(f"\nEvaluating {model_name}...")
        
        # Predictions
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        
        # Calculate metrics
        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred, zero_division=0),
            'recall': recall_score(y_test, y_pred, zero_division=0),
            'f1': f1_score(y_test, y_pred, zero_division=0),
            'roc_auc': roc_auc_score(y_test, y_pred_proba),
            'pr_auc': average_precision_score(y_test, y_pred_proba)
        }
        
        # Print metrics
        print(f"Accuracy:  {metrics['accuracy']:.4f}")
        print(f"Precision: {metrics['precision']:.4f}")
        print(f"Recall:    {metrics['recall']:.4f}")
        print(f"F1-Score:  {metrics['f1']:.4f}")
        print(f"ROC-AUC:   {metrics['roc_auc']:.4f}")
        print(f"PR-AUC:    {metrics['pr_auc']:.4f}")
        
        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        print("\nConfusion Matrix:")
        print(cm)
        
        # Classification report
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred, target_names=['Legitimate', 'Fraud']))
        
        # Store results
        self.results[model_name] = {
            'metrics': metrics,
            'confusion_matrix': cm,
            'y_pred': y_pred,
            'y_pred_proba': y_pred_proba
        }
        
        return metrics, cm
    
    def cross_validate(self, model, X, y, cv=5, model_name=""):
        """Perform cross-validation"""
        print(f"\nPerforming {cv}-fold cross-validation for {model_name}...")
        
        skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=self.random_state)
        
        cv_scores = {
            'accuracy': [],
            'precision': [],
            'recall': [],
            'f1': [],
            'roc_auc': []
        }
        
        for fold, (train_idx, val_idx) in enumerate(skf.split(X, y), 1):
            X_train_fold, X_val_fold = X.iloc[train_idx], X.iloc[val_idx]
            y_train_fold, y_val_fold = y.iloc[train_idx], y.iloc[val_idx]
            
            # Handle imbalance for this fold
            X_train_res, y_train_res = self.handle_imbalance(X_train_fold, y_train_fold)
            
            # Train and predict
            model.fit(X_train_res, y_train_res)
            y_pred = model.predict(X_val_fold)
            y_pred_proba = model.predict_proba(X_val_fold)[:, 1]
            
            # Calculate scores
            cv_scores['accuracy'].append(accuracy_score(y_val_fold, y_pred))
            cv_scores['precision'].append(precision_score(y_val_fold, y_pred, zero_division=0))
            cv_scores['recall'].append(recall_score(y_val_fold, y_pred, zero_division=0))
            cv_scores['f1'].append(f1_score(y_val_fold, y_pred, zero_division=0))
            cv_scores['roc_auc'].append(roc_auc_score(y_val_fold, y_pred_proba))
            
            print(f"  Fold {fold}: ROC-AUC = {cv_scores['roc_auc'][-1]:.4f}")
        
        # Print CV results
        print(f"\nCross-Validation Results for {model_name}:")
        for metric, scores in cv_scores.items():
            mean_score = np.mean(scores)
            std_score = np.std(scores)
            print(f"  {metric}: {mean_score:.4f} (+/- {std_score:.4f})")
        
        return cv_scores
    
    def compare_models(self):
        """Compare all trained models"""
        if not self.results:
            print("No results to compare. Train models first.")
            return None
        
        print("\n" + "="*50)
        print("MODEL COMPARISON")
        print("="*50)
        
        # Create comparison dataframe
        comparison_data = []
        for model_name, result in self.results.items():
            metrics = result['metrics']
            row = {
                'Model': model_name,
                'Accuracy': f"{metrics['accuracy']:.4f}",
                'Precision': f"{metrics['precision']:.4f}",
                'Recall': f"{metrics['recall']:.4f}",
                'F1-Score': f"{metrics['f1']:.4f}",
                'ROC-AUC': f"{metrics['roc_auc']:.4f}",
                'PR-AUC': f"{metrics['pr_auc']:.4f}"
            }
            comparison_data.append(row)
        
        comparison_df = pd.DataFrame(comparison_data)
        print(comparison_df.to_string(index=False))
        
        # Determine best model based on PR-AUC (most important for imbalanced data)
        best_model_name = max(self.results.items(), 
                            key=lambda x: x[1]['metrics']['pr_auc'])[0]
        
        self.best_model = self.models[best_model_name]
        self.best_model_name = best_model_name
        
        print(f"\nBest Model: {best_model_name}")
        print(f"Best PR-AUC: {self.results[best_model_name]['metrics']['pr_auc']:.4f}")
        
        # Plot comparison
        self.plot_model_comparison(comparison_df)
        
        return comparison_df
    
    def plot_model_comparison(self, comparison_df):
        """Plot model comparison"""
        plt.figure(figsize=(15, 10))
        
        metrics_to_plot = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC-AUC', 'PR-AUC']
        
        # Convert string metrics to float for plotting
        plot_data = comparison_df.copy()
        for metric in metrics_to_plot:
            plot_data[metric] = plot_data[metric].astype(float)
        
        # Create subplots
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        axes = axes.flatten()
        
        for idx, metric in enumerate(metrics_to_plot):
            ax = axes[idx]
            bars = ax.bar(plot_data['Model'], plot_data[metric])
            ax.set_title(f'{metric} Comparison', fontsize=14)
            ax.set_ylabel(metric)
            ax.tick_params(axis='x', rotation=45)
            
            # Highlight best model
            if metric == 'PR-AUC':
                best_idx = plot_data[metric].idxmax()
                bars[best_idx].set_color('red')
            
            # Add value labels
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                       f'{height:.3f}', ha='center', va='bottom', fontsize=9)
        
        plt.suptitle('Model Performance Comparison', fontsize=16, y=1.02)
        plt.tight_layout()
        
        # Save plot
        os.makedirs("../notebooks/visualizations", exist_ok=True)
        plt.savefig("../notebooks/visualizations/model_comparison.png", 
                   dpi=150, bbox_inches='tight')
        plt.show()
    
    def save_best_model(self, output_dir="../models"):
        """Save the best model"""
        if self.best_model is None:
            print("No best model to save. Train models first.")
            return
        
        os.makedirs(output_dir, exist_ok=True)
        model_path = f"{output_dir}/best_model_{self.best_model_name}.pkl"
        
        joblib.dump(self.best_model, model_path)
        print(f"\nBest model saved to: {model_path}")
        
        # Also save results
        results_path = f"{output_dir}/model_results.csv"
        comparison_df = self.compare_models()
        if comparison_df is not None:
            comparison_df.to_csv(results_path, index=False)
            print(f"Model results saved to: {results_path}")
        
        return model_path

def run_model_training():
    """Run complete model training pipeline"""
    print("="*60)
    print("FRAUD DETECTION - MODEL TRAINING PIPELINE")
    print("="*60)
    
    # Initialize trainer
    trainer = FraudModelTrainer(random_state=42)
    
    # Load data
    X, y, df = trainer.load_data()
    
    # Prepare data split
    X_train, X_test, y_train, y_test = trainer.prepare_data(X, y, test_size=0.3)
    
    # Handle imbalance in training data
    X_train_res, y_train_res = trainer.handle_imbalance(X_train, y_train, method='smote')
    
    # Train models
    print("\n" + "="*60)
    print("TRAINING MODELS")
    print("="*60)
    
    # Logistic Regression
    lr_model = trainer.train_logistic_regression(X_train_res, y_train_res)
    trainer.evaluate_model(lr_model, X_test, y_test, 'logistic_regression')
    
    # Random Forest
    rf_model = trainer.train_random_forest(X_train_res, y_train_res)
    trainer.evaluate_model(rf_model, X_test, y_test, 'random_forest')
    
    # XGBoost
    xgb_model = trainer.train_xgboost(X_train_res, y_train_res)
    trainer.evaluate_model(xgb_model, X_test, y_test, 'xgboost')
    
    # LightGBM (optional)
    try:
        lgb_model = trainer.train_lightgbm(X_train_res, y_train_res)
        trainer.evaluate_model(lgb_model, X_test, y_test, 'lightgbm')
    except Exception as e:
        print(f"LightGBM training failed: {e}")
    
    # Compare models
    comparison_df = trainer.compare_models()
    
    # Save best model
    model_path = trainer.save_best_model()
    
    # Generate final report
    generate_training_report(trainer, X_test, y_test)
    
    return trainer

def generate_training_report(trainer, X_test, y_test):
    """Generate training report"""
    print("\n" + "="*60)
    print("TRAINING REPORT")
    print("="*60)
    
    if trainer.best_model is None:
        print("No best model identified.")
        return
    
    print(f"\nBest Model: {trainer.best_model_name}")
    best_metrics = trainer.results[trainer.best_model_name]['metrics']
    
    print("\nPerformance Metrics:")
    print(f"  Accuracy:  {best_metrics['accuracy']:.4f}")
    print(f"  Precision: {best_metrics['precision']:.4f}")
    print(f"  Recall:    {best_metrics['recall']:.4f}")
    print(f"  F1-Score:  {best_metrics['f1']:.4f}")
    print(f"  ROC-AUC:   {best_metrics['roc_auc']:.4f}")
    print(f"  PR-AUC:    {best_metrics['pr_auc']:.4f}")
    
    # Get feature importance for tree-based models
    if hasattr(trainer.best_model, 'feature_importances_'):
        feature_names = X_test.columns
        importances = trainer.best_model.feature_importances_
        feature_importance = pd.DataFrame({
            'feature': feature_names,
            'importance': importances
        }).sort_values('importance', ascending=False)
        
        print(f"\nTop 10 Most Important Features:")
        print(feature_importance.head(10).to_string(index=False))
    
    print(f"\nModel saved to: ../models/best_model_{trainer.best_model_name}.pkl")
    print("\nNext Steps:")
    print("1. Run SHAP analysis for model explainability")
    print("2. Test the model on new data")
    print("3. Deploy the model for real-time predictions")

if __name__ == "__main__":
    # Run training pipeline
    trainer = run_model_training()
    
    print("\n" + "="*60)
    print("MODEL TRAINING COMPLETED!")
    print("="*60)