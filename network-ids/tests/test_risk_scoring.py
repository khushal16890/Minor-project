"""
Unit tests for Threat Severity & Risk Assessment Engine.
"""
import pytest
from mitigation.risk_engine import RiskEngine
from backend.core.config import RiskWeights


def test_risk_score_bounds_and_severity():
    engine = RiskEngine()

    # Benign case: 0 confidence, normal attack, 0 anomaly
    benign_eval = engine.assess(model_confidence=0.0, attack_type="benign", anomaly_score=0.0, frequency_score=0.0)
    assert 0.0 <= benign_eval.risk_score <= 29.0
    assert benign_eval.severity_level == "low"

    # Critical DoS case: high confidence, severe attack, high anomaly
    crit_eval = engine.assess(model_confidence=0.95, attack_type="dos", anomaly_score=0.90, frequency_score=10.0)
    assert 60.0 <= crit_eval.risk_score <= 100.0
    assert crit_eval.severity_level in ["high", "critical"]
    assert crit_eval.recommended_response != ""


def test_custom_weights():
    weights = RiskWeights(model_confidence=0.70, attack_severity=0.10, anomaly_score=0.10, frequency_score=0.10)
    engine = RiskEngine(weights=weights)

    eval_res = engine.assess(model_confidence=0.90, attack_type="benign", anomaly_score=0.0, frequency_score=0.0)
    assert eval_res.risk_score >= 60.0
