# scripts/eda_and_feature_engineering.py

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from src.data_processing import EcommerceFraudData

sns.set(style="whitegrid")

# ----------------------------
# File paths
# ----------------------------
FRAUD_FILE = "data/raw/Fraud_Data.csv"
IP_FILE = "data/raw/IpAddress_to_Country.csv"

# ----------------------------
# Load and clean data
# ----------------------------
ecom_data = EcommerceFraudData(fraud_filepath=FRAUD_FILE, ip_filepath=IP_FILE)
df = ecom_data.load_data()
df = ecom_data.clean_data()

# ----------------------------
# Map IP to country
# ----------------------------
df = ecom_data.merge_ip_country()

# ----------------------------
# Feature engineering
# ----------------------------
df = ecom_data.feature_engineering()

# ----------------------------
# Visualizations - EDA
# ----------------------------
def plot_class_distribution(df):
    sns.countplot(x='class', data=df)
    plt.title('Class Distribution (0=Legit, 1=Fraud)')
    plt.show()

def plot_country_fraud(df):
    country_fraud = df.groupby('country')['class'].mean().sort_values(ascending=False)
    country_fraud.plot(kind='bar', figsize=(12, 6))
    plt.ylabel("Fraud Rate")
    plt.title("Fraud Rate by Country")
    plt.show()

def plot_time_features(df):
    sns.histplot(df['hour_of_day'], bins=24, kde=False)
    plt.title("Transactions by Hour of Day")
    plt.show()

    sns.histplot(df['day_of_week'], bins=7, kde=False)
    plt.title("Transactions by Day of Week")
    plt.show()

# Run visualizations
plot_class_distribution(df)
plot_country_fraud(df)
plot_time_features(df)

# ----------------------------
# Preprocessing for modeling
# ----------------------------
df_processed = ecom_data.preprocess_features()

# ----------------------------
# Save processed dataset
# ----------------------------
df_processed.to_csv("data/processed/fraud_data_processed.csv", index=False)
print("Processed dataset saved to data/processed/fraud_data_processed.csv")
