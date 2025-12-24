# scripts/run_pipeline.py

import os
from src.data_processing import EcommerceFraudData

def main():
    """
    Main pipeline to process and feature-engineer the fraud datasets.
    This script orchestrates loading, cleaning, feature engineering, 
    EDA visualization, and saving processed datasets for modeling.
    """

    # ----------------------------
    # File paths
    # ----------------------------
    fraud_file = os.path.join("data", "raw", "Fraud_Data.csv")
    ip_file = os.path.join("data", "raw", "IpAddress_to_Country.csv")
    output_file = os.path.join("data", "processed", "fraud_data_processed.csv")

    # ----------------------------
    # Initialize Data Processor
    # ----------------------------
    ecom_data = EcommerceFraudData(fraud_filepath=fraud_file, ip_filepath=ip_file)

    # ----------------------------
    # Load and clean data
    # ----------------------------
    print("[INFO] Loading and cleaning data...")
    df = ecom_data.load_data()
    df = ecom_data.clean_data()

    # ----------------------------
    # Map IP to Country
    # ----------------------------
    print("[INFO] Mapping IP addresses to countries...")
    df = ecom_data.merge_ip_country()

    # ----------------------------
    # Feature Engineering
    # ----------------------------
    print("[INFO] Creating features...")
    df = ecom_data.feature_engineering()

    # ----------------------------
    # EDA Visualizations (optional)
    # ----------------------------
    print("[INFO] Running EDA visualizations...")
    try:
        import scripts.eda_and_feature_engineering as eda_module
        eda_module.plot_class_distribution(df)
        eda_module.plot_country_fraud(df)
        eda_module.plot_time_features(df)
    except ImportError:
        print("[WARNING] EDA module not found. Skipping plots.")

    # ----------------------------
    # Preprocess features for modeling
    # ----------------------------
    print("[INFO] Preprocessing features for modeling...")
    df_processed = ecom_data.preprocess_features()

    # ----------------------------
    # Save processed dataset
    # ----------------------------
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    df_processed.to_csv(output_file, index=False)
    print(f"[INFO] Processed dataset saved to {output_file}")

if __name__ == "__main__":
    main()
