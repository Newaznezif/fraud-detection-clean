# Fraud Detection Project

## Business Context
Financial fraud is a major concern for businesses and consumers alike. This project aims to detect fraudulent transactions in two datasets—`Fraud_Data.csv` and `creditcard.csv`—using machine learning models. Accurate fraud detection helps reduce financial losses, mitigate risk, and improve trust in financial systems.

---

## Repository Structure
fraud-detection/
├── data/ # Raw datasets (Fraud_Data.csv, creditcard.csv)
├── models/ # Saved trained models (e.g., Random Forest)
├── notebooks/ # Jupyter notebooks for exploration and modeling
│ └── modeling.ipynb
├── reports/ # Generated reports, evaluation metrics, plots
├── scripts/ # Python scripts to run training and evaluation
│ └── run_task2.py
├── src/ # Source code modules
│ └── task2_models.py
├── tests/ # Unit and integration tests
├── requirements.txt # Project dependencies
└── README.md # Project documentation

yaml
Copy code

---

## Setup Instructions

1. **Clone the repository**
```bash
git clone https://github.com/Newaznezif/fraud-detection-clean.git
cd fraud-detection-clean
Install dependencies

bash
Copy code
pip install -r requirements.txt
Prepare data

Place Fraud_Data.csv and creditcard.csv inside data/raw/.

How to Run
Task 2 – Model Building and Training
This task trains and evaluates machine learning models for fraud detection.

bash
Copy code
python scripts/run_task2.py
What it does:

Loads both datasets

Preprocesses the data (including time-based features for Fraud_Data.csv)

Performs stratified train/test split

Trains a Logistic Regression baseline model

Trains a Random Forest ensemble model with basic hyperparameter tuning

Performs 5-fold stratified cross-validation

Compares models and selects the best one

Saves the best model in the models/ directory

Task 3 – Model Explainability
(Optional if implemented)

Provides SHAP or feature importance explanations for the selected model.

Evaluation Metrics
All models are evaluated using:

PR-AUC (Precision-Recall AUC) – suitable for imbalanced datasets

ROC-AUC

F1 Score

Confusion Matrix

Cross-validation reports mean and standard deviation of metrics.

Recommendations for Users
Ensure Python ≥ 3.10

Verify that all dependencies in requirements.txt are installed

Always run tasks in order: Task 2 → Task 3

Use reports/ folder to save plots or exported notebooks for documentation

Notes
The repository supports both Fraud_Data.csv and creditcard.csv datasets.

Models are trained with class imbalance handling (class_weight='balanced').

Hyperparameter tuning is lightweight and can be extended for production.

Contributing
Fork the repo

Create a new branch: git checkout -b feature/your-feature

Make your changes and commit: git commit -m "Add new feature"

Push to branch: git push origin feature/your-feature

Open a Pull Request

License
MIT License. See LICENSE for details.

Contact
Newaz Nezif – GitHub
Email: newaz@example.com

yaml
Copy code

---

# Task 3: Model Explainability - Summary Report

## ✅ Requirements Completed

### 1. Feature Importance Baseline
- Extracted built-in Random Forest feature importance
- Visualized top 10 features with gradient coloring
- Built-in importance shows `hours_since_signup` dominates (82.3%)

### 2. SHAP Analysis (Permutation Importance)
- Implemented SHAP-like permutation importance
- Generated summary plot showing global feature importance
- Permutation importance reveals only `hours_since_signup` matters (9.4%)

### 3. Individual Predictions Analysis
- Found and analyzed TP, FP, FN prediction cases
- Created force plots showing feature contributions
- Saved visualizations to `reports/task3_explainability/force_plots/`

### 4. Interpretation & Insights
- Compared built-in vs permutation importance methods
- Identified top 5 fraud drivers (dominated by `hours_since_signup`)
- Documented counterintuitive findings

### 5. Business Recommendations
- Generated 8 actionable recommendations
- Prioritized by impact and effort
- Saved to `reports/recommendations/business_recommendations.csv`

## 🔍 Critical Finding

**Feature Dominance Issue:** `hours_since_signup` accounts for 82.3% of built-in importance and 100% of permutation importance. This suggests:
- Possible data leakage
- Model overfitting to a single feature
- Need for feature engineering diversity

## 📊 Files Generated

- `notebooks/shap-explainability.ipynb` - Complete analysis notebook
- `reports/task3_explainability/` - All analysis outputs
- `notebooks/reports/` - Visualization and recommendation files
- `models/random_forest.pkl` - Trained model

## 🚨 Recommendations for Improvement

1. **Investigate Data Quality**: Check if `hours_since_signup` has data leakage
2. **Feature Engineering**: Create additional time-based features
3. **Model Robustness**: Test model without the dominant feature
4. **Threshold Optimization**: Fine-tune based on business costs