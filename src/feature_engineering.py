# src/feature_engineering.py

import pandas as pd
import numpy as np
import ipaddress

# ================================
# Transaction-related features
# ================================
def add_transaction_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add transaction-related features:
    - purchase_hour
    - purchase_day
    - signup_to_purchase_days
    - log_purchase_value
    """
    df['signup_time'] = pd.to_datetime(df['signup_time'])
    df['purchase_time'] = pd.to_datetime(df['purchase_time'])

    df['purchase_hour'] = df['purchase_time'].dt.hour
    df['purchase_day'] = df['purchase_time'].dt.dayofweek
    df['signup_to_purchase_days'] = (
        (df['purchase_time'] - df['signup_time']).dt.total_seconds() / (24*3600)
    )
    df['log_purchase_value'] = np.log1p(df['purchase_value'])

    return df

# ================================
# IP Address conversion
# ================================
def safe_ip_to_int(ip: str) -> int | None:
    """
    Convert IPv4 string to integer safely.
    Returns None if conversion fails.
    """
    try:
        return int(ipaddress.IPv4Address(ip))
    except (ValueError, TypeError):
        return None

def add_ip_int_feature(df: pd.DataFrame, ip_col: str = 'ip_address') -> pd.DataFrame:
    """
    Add numeric 'ip_int' column to the dataframe.
    """
    df[ip_col] = df[ip_col].astype(str)
    df['ip_int'] = df[ip_col].apply(safe_ip_to_int)
    return df

# ================================
# Categorical encoding
# ================================
def encode_categorical_features(df: pd.DataFrame, columns: list) -> pd.DataFrame:
    """
    Factorize categorical columns into numeric codes.
    """
    for col in columns:
        if col in df.columns:
            df[f"{col}_code"] = pd.factorize(df[col])[0]
    return df

# ================================
# One-hot encoding
# ================================
def one_hot_encode(df: pd.DataFrame, column: str, drop_first: bool = True) -> pd.DataFrame:
    """
    One-hot encode a categorical column.
    Missing values are filled with 'Unknown'.
    """
    if column in df.columns:
        df[column] = df[column].fillna('Unknown')
        df = pd.get_dummies(df, columns=[column], prefix=column, drop_first=drop_first)
    return df

# ================================
# Example usage
# ================================
if __name__ == "__main__":
    df = pd.read_csv("../data/raw/Fraud_Data.csv")
    df = add_transaction_features(df)
    df = add_ip_int_feature(df)
    df = encode_categorical_features(df, columns=['browser', 'device_id'])
    df = one_hot_encode(df, 'country')
    print(df.head())
