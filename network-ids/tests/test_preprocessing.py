"""
Unit tests for data preprocessing pipeline.
"""
import pytest
import numpy as np
import pandas as pd
from detection.preprocessing.pipeline import PreprocessingPipeline


def test_pipeline_clean_and_fit(sample_df):
    pipeline = PreprocessingPipeline()
    # Insert NaN and infinite values to verify cleaning
    sample_df = sample_df.copy()
    sample_df.loc[0, "flow_duration"] = np.nan

    X_mat, y_vec = pipeline.fit_transform(sample_df, label_col="Label")

    assert not np.isnan(X_mat).any(), "Cleaned transformed matrix must not contain any NaNs"
    assert not np.isinf(X_mat).any(), "Cleaned transformed matrix must not contain infs"
    assert pipeline.is_fitted is True


def test_pipeline_no_data_leakage(sample_df):
    # Add identifier columns to verify they get removed
    sample_df = sample_df.copy()
    sample_df["src_ip"] = "192.168.1.100"
    sample_df["dst_ip"] = "10.0.0.1"
    sample_df["timestamp"] = "2026-11-24 12:00:00"
    sample_df["flow_id"] = "flow-999"

    pipeline = PreprocessingPipeline()
    X_mat, y_vec = pipeline.fit_transform(sample_df, label_col="Label")

    # Identifiers must not be present in feature_names
    for col in ["src_ip", "dst_ip", "timestamp", "flow_id"]:
        assert col not in pipeline.feature_names, f"Identifier {col} leaked into feature matrix"


def test_pipeline_transform_consistency(sample_df):
    pipeline = PreprocessingPipeline()
    X_train, _ = pipeline.fit_transform(sample_df.iloc[:80], label_col="Label")
    X_test = pipeline.transform(sample_df.iloc[80:])

    assert X_train.shape[1] == X_test.shape[1], "Train and test feature spaces must be identical dimension"
    assert len(pipeline.feature_names) == X_train.shape[1]
