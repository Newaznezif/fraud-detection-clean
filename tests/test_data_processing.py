import numpy as np
import pandas as pd
import pytest

from src.imbalance_handler import handle_imbalance
from src.feature_engineering import safe_ip_to_int
def test_safe_ip_to_int_valid_ip():
    ip = "192.168.1.1"
    result = safe_ip_to_int(ip)
    assert isinstance(result, int)
    assert result > 0
def test_safe_ip_to_int_invalid_ip():
    ip = "invalid_ip"
    result = safe_ip_to_int(ip)
    assert result is None
def test_handle_imbalance_smote():
    X = np.random.rand(100, 5)
    y = np.array([0] * 95 + [1] * 5)

    X_res, y_res = handle_imbalance(
        X,
        y,
        strategy="smote",
        sampling_ratio=0.5,
        random_state=42
    )

    unique, counts = np.unique(y_res, return_counts=True)
    class_dist = dict(zip(unique, counts))

    assert class_dist[1] > 5
    assert len(X_res) == len(y_res)
def test_handle_imbalance_none():
    X = np.random.rand(50, 3)
    y = np.array([0] * 45 + [1] * 5)

    X_res, y_res = handle_imbalance(X, y, strategy="none")

    assert np.array_equal(X, X_res)
    assert np.array_equal(y, y_res)
def test_handle_imbalance_invalid_strategy():
    X = np.random.rand(10, 2)
    y = np.array([0, 1] * 5)

    with pytest.raises(ValueError):
        handle_imbalance(X, y, strategy="magic")
