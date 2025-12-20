"""
Unit tests for data processing module
"""
import pytest
import pandas as pd
import numpy as np
from datetime import datetime
import sys
import os

# Add src to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data_processing import FraudDataProcessor

class TestFraudDataProcessor:
    """Test suite for FraudDataProcessor"""
    
    @pytest.fixture
    def sample_fraud_data(self):
        """Create sample fraud data for testing"""
        data = {
            'user_id': [1, 2, 3, 4, 5],
            'signup_time': ['2023-01-01 10:00:00', '2023-01-01 11:00:00', 
                           '2023-01-01 12:00:00', '2023-01-01 13:00:00', 
                           '2023-01-01 14:00:00'],
            'purchase_time': ['2023-01-01 10:30:00', '2023-01-01 11:30:00',
                             '2023-01-01 12:30:00', '2023-01-01 13:30:00',
                             '2023-01-01 14:30:00'],
            'purchase_value': [100.0, 200.0, 300.0, 400.0, 500.0],
            'device_id': ['device1', 'device2', 'device3', 'device1', 'device2'],
            'source': ['SEO', 'Ads', 'Direct', 'SEO', 'Ads'],
            'browser': ['Chrome', 'Firefox', 'Chrome', 'Safari', 'Chrome'],
            'sex': ['M', 'F', 'M', 'F', 'M'],
            'age': [25, 30, 35, 40, 45],
            'ip_address': ['192.168.1.1', '192.168.1.2', '192.168.1.3', 
                          '192.168.1.4', '192.168.1.5'],
            'class': [0, 0, 1, 0, 1]
        }
        return pd.DataFrame(data)
    
    @pytest.fixture
    def sample_ip_country_data(self):
        """Create sample IP country data for testing"""
        data = {
            'lower_bound_ip_address': [3232235776, 3232235876, 3232235976],
            'upper_bound_ip_address': [3232235875, 3232235975, 3232236075],
            'country': ['United States', 'United Kingdom', 'Canada']
        }
        return pd.DataFrame(data)
    
    def test_ip_to_int_conversion(self):
        """Test IP address to integer conversion"""
        processor = FraudDataProcessor()
        
        # Test valid IP
        assert processor.ip_to_int('192.168.1.1') == 3232235777
        
        # Test invalid IP
        assert np.isnan(processor.ip_to_int('invalid'))
        assert np.isnan(processor.ip_to_int('192.168.1'))
        
        # Test None/NaN
        assert np.isnan(processor.ip_to_int(None))
        assert np.isnan(processor.ip_to_int(np.nan))
    
    def test_load_and_clean_fraud_data(self, sample_fraud_data, tmp_path):
        """Test data loading and cleaning"""
        # Save sample data to temp file
        test_file = tmp_path / "test_fraud_data.csv"
        sample_fraud_data.to_csv(test_file, index=False)
        
        processor = FraudDataProcessor()
        df = processor.load_and_clean_fraud_data(str(test_file))
        
        # Check data types
        assert df['signup_time'].dtype == 'datetime64[ns]'
        assert df['purchase_time'].dtype == 'datetime64[ns]'
        
        # Check no missing values in critical columns
        assert df['age'].isnull().sum() == 0
        assert df['browser'].isnull().sum() == 0
        assert df['source'].isnull().sum() == 0
        
        # Check shape
        assert df.shape[0] == 5
        assert df.shape[1] >= 11  # Original columns
    
    def test_create_time_features(self, sample_fraud_data):
        """Test time feature creation"""
        processor = FraudDataProcessor()
        
        # Convert timestamps first
        sample_fraud_data['signup_time'] = pd.to_datetime(sample_fraud_data['signup_time'])
        sample_fraud_data['purchase_time'] = pd.to_datetime(sample_fraud_data['purchase_time'])
        
        df = processor.create_time_features(sample_fraud_data)
        
        # Check new columns
        assert 'purchase_hour' in df.columns
        assert 'purchase_day_of_week' in df.columns
        assert 'hours_since_signup' in df.columns
        assert 'time_of_day' in df.columns
        assert 'is_weekend' in df.columns
        
        # Check values
        assert all(0 <= hour <= 23 for hour in df['purchase_hour'])
        assert all(0 <= day <= 6 for day in df['purchase_day_of_week'])
        assert all(df['hours_since_signup'] >= 0)
    
    def test_create_user_behavior_features(self, sample_fraud_data):
        """Test user behavior feature creation"""
        processor = FraudDataProcessor()
        
        # Convert timestamps
        sample_fraud_data['signup_time'] = pd.to_datetime(sample_fraud_data['signup_time'])
        sample_fraud_data['purchase_time'] = pd.to_datetime(sample_fraud_data['purchase_time'])
        
        df = processor.create_user_behavior_features(sample_fraud_data)
        
        # Check user statistics columns
        assert 'user_transaction_count' in df.columns
        assert 'avg_transaction_amount' in df.columns
        assert 'total_spent' in df.columns
        
        # Check derived columns
        assert 'transaction_number' in df.columns
        assert 'time_since_last_transaction' in df.columns
        assert 'amount_to_avg_ratio' in df.columns
        
        # Check calculations
        user_1_data = df[df['user_id'] == 1]
        assert user_1_data['user_transaction_count'].iloc[0] == 1
        assert user_1_data['avg_transaction_amount'].iloc[0] == 100.0
    
    def test_prepare_features_pipeline(self, sample_fraud_data, sample_ip_country_data):
        """Test complete feature engineering pipeline"""
        processor = FraudDataProcessor()
        processor.ip_country_df = sample_ip_country_data
        
        # Convert timestamps
        sample_fraud_data['signup_time'] = pd.to_datetime(sample_fraud_data['signup_time'])
        sample_fraud_data['purchase_time'] = pd.to_datetime(sample_fraud_data['purchase_time'])
        
        df = processor.prepare_features(sample_fraud_data)
        
        # Check that many features were created
        assert df.shape[1] > 20  # Should have many new features
        
        # Check specific feature types
        assert any('_encoded' in col for col in df.columns)  # Encoded features
        assert any('risk' in col.lower() for col in df.columns)  # Risk features
        
        # Check no NaN in critical features
        critical_cols = ['hours_since_signup', 'transaction_number']
        for col in critical_cols:
            if col in df.columns:
                assert df[col].isnull().sum() == 0
    
    def test_save_processed_data(self, sample_fraud_data, tmp_path):
        """Test saving processed data"""
        processor = FraudDataProcessor()
        
        output_file = tmp_path / "processed_data.csv"
        
        # Process and save
        df = processor.create_time_features(sample_fraud_data)
        saved_path = processor.save_processed_data(df, str(output_file))
        
        # Check file was created
        assert output_file.exists()
        
        # Check can be loaded back
        loaded_df = pd.read_csv(saved_path)
        assert loaded_df.shape == df.shape
        assert list(loaded_df.columns) == list(df.columns)

if __name__ == "__main__":
    pytest.main([__file__, "-v"])