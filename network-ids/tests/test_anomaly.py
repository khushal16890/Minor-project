"""
Unit tests for Isolation Forest anomaly detection engine.
"""
import pytest
import pandas as pd
from detection.anomaly.isolation_forest import AnomalyDetector


def test_isolation_forest_train_and_predict(sample_df):
    detector = AnomalyDetector(contamination=0.1)
    X = sample_df.drop(columns=["Label"])

    # Train on benign samples only
    benign_X = X.iloc[:50]
    train_res = detector.train(benign_X)
    assert "model_name" in train_res
    assert train_res["n_train_samples"] > 0

    # Single sample prediction
    single_res = detector.predict_single(X.iloc[0].to_dict())
    assert "anomaly_score" in single_res
    assert "is_anomaly" in single_res
    assert isinstance(single_res["is_anomaly"], (bool, int))
