# src/data_processing.py

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from typing import Optional

# ----------------------------
# E-Commerce Fraud Data Class
# ----------------------------
class EcommerceFraudData:
    def __init__(self, fraud_filepath: str, ip_filepath: str):
        """
        Initialize with file paths for fraud data and IP-to-country mapping
        """
        self.fraud_filepath = fraud_filepath
        self.ip_filepath = ip_filepath
        self.df: Optional[pd.DataFrame] = None
        self.ip_df: Optional[pd.DataFrame] = None

    def load_data(self) -> pd.DataFrame:
        """Load raw data."""
        self.df = pd.read_csv(self.fraud_filepath)
        self.ip_df = pd.read_csv(self.ip_filepath)
        return self.df

    def clean_data(self) -> pd.DataFrame:
        """Remove duplicates, handle missing values, and convert data types."""
        df = self.df.drop_duplicates()
        df['signup_time'] = pd.to_datetime(df['signup_time'])
        df['purchase_time'] = pd.to_datetime(df['purchase_time'])

        # Impute categorical columns
        for col in ['sex', 'source', 'browser', 'device_id']:
            if df[col].isnull().sum() > 0:
                df[col] = df[col].fillna(df[col].mode()[0])

        # Impute numeric columns
        df['age'] = df['age'].fillna(df['age'].median())
        df['purchase_value'] = df['purchase_value'].fillna(df['purchase_value'].median())

        self.df = df
        return self.df

    @staticmethod
    def ip_to_int(ip: str) -> int:
        """Convert IPv4 string to integer."""
        parts = ip.split('.')
        return sum(int(part) << (8 * (3 - i)) for i, part in enumerate(parts))

    def merge_ip_country(self) -> pd.DataFrame:
        """Map IP addresses to countries using range-based lookup."""
        df = self.df.copy()
        ip_df = self.ip_df.copy()

        # Convert IPs to integer format
        df['ip_int'] = df['ip_address'].apply(self.ip_to_int)
        ip_df['lower_bound_ip_address'] = ip_df['lower_bound_ip_address'].astype(int)
        ip_df['upper_bound_ip_address'] = ip_df['upper_bound_ip_address'].astype(int)

        # Create interval index for fast range lookup
        intervals = pd.IntervalIndex.from_arrays(
            ip_df['lower_bound_ip_address'], ip_df['upper_bound_ip_address'], closed='both'
        )

        def get_country(ip_int):
            try:
                idx = intervals.get_loc(ip_int)
                return ip_df['country'].iloc[idx]
            except KeyError:
                return 'Unknown'

        df['country'] = df['ip_int'].apply(get_country)
        self.df = df
        return self.df

    def feature_engineering(self) -> pd.DataFrame:
        """Add time-based and user activity features."""
        df = self.df.copy()
        df['hour_of_day'] = df['purchase_time'].dt.hour
        df['day_of_week'] = df['purchase_time'].dt.dayofweek
        df['time_since_signup'] = (df['purchase_time'] - df['signup_time']).dt.total_seconds()
        df['transactions_per_user'] = df.groupby('user_id')['purchase_time'].transform('count')
        self.df = df
        return self.df

    def preprocess_features(self) -> pd.DataFrame:
        """Scale numeric features and encode categorical features."""
        numeric_cols = ['purchase_value', 'age', 'time_since_signup', 'transactions_per_user', 'hour_of_day', 'day_of_week']
        categorical_cols = ['sex', 'source', 'browser', 'country']

        df = self.df.copy()

        # Scale numeric features
        scaler = StandardScaler()
        df[numeric_cols] = scaler.fit_transform(df[numeric_cols])

        # One-hot encode categorical features
        df = pd.get_dummies(df, columns=categorical_cols, drop_first=True)

        self.df = df
        return self.df

# ----------------------------
# Credit Card Fraud Data Class
# ----------------------------
class CreditCardFraudData:
    def __init__(self, filepath: str):
        self.filepath = filepath
        self.df: Optional[pd.DataFrame] = None

    def load_data(self) -> pd.DataFrame:
        """Load raw credit card dataset."""
        self.df = pd.read_csv(self.filepath)
        return self.df

    def clean_data(self) -> pd.DataFrame:
        """Remove duplicates and handle missing values."""
        df = self.df.drop_duplicates()
        numeric_cols = ['Time', 'Amount'] + [f'V{i}' for i in range(1, 29)]
        for col in numeric_cols:
            if df[col].isnull().sum() > 0:
                df[col] = df[col].fillna(df[col].median())
        self.df = df
        return self.df

    def preprocess_features(self) -> pd.DataFrame:
        """Scale numeric features."""
        numeric_cols = ['Time', 'Amount'] + [f'V{i}' for i in range(1, 29)]
        df = self.df.copy()
        scaler = StandardScaler()
        df[numeric_cols] = scaler.fit_transform(df[numeric_cols])
        self.df = df
        return self.df
