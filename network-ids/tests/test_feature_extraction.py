"""
Unit tests for network feature extraction.
"""
import pytest
from detection.feature_extraction.extractor import FeatureExtractor
from backend.core.config import FEATURE_SCHEMA


def test_extractor_schema_completeness(sample_feature_dict):
    extractor = FeatureExtractor()
    # Test that extraction produces valid numeric features
    features = extractor.extract_from_dict(sample_feature_dict)

    assert isinstance(features, dict)
    # Check that crucial network metrics are present
    crucial_features = [
        "flow_duration", "fwd_packets", "bwd_packets",
        "total_fwd_bytes", "total_bwd_bytes", "flow_bytes_per_sec",
        "flow_pkts_per_sec", "avg_pkt_size"
    ]
    for feat in crucial_features:
        assert feat in features, f"Crucial feature {feat} missing from extractor output"
        assert isinstance(features[feat], (int, float)), f"Feature {feat} must be numeric"


def test_extractor_zero_duration_safe():
    extractor = FeatureExtractor()
    bad_flow = {"flow_duration": 0.0, "total_fwd_bytes": 100, "fwd_packets": 2}
    features = extractor.extract_from_dict(bad_flow)

    assert not any(features.get(k) == float("inf") for k in ["flow_bytes_per_sec", "flow_pkts_per_sec"]), "Zero duration must not cause infinity"
