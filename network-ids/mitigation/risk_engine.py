"""
Risk Scoring Engine — quantify threat risk and recommend responses.

Computes a composite risk score (0–100) using the formula::

    Risk = w_mc × ModelConfidence
         + w_as × AttackSeverity
         + w_an × AnomalyScore
         + w_fr × FrequencyScore

Default weights (from ``RiskWeights``):
    model_confidence 0.40, attack_severity 0.30,
    anomaly_score 0.20, frequency_score 0.10

Risk levels:
    Low 0–29  |  Medium 30–59  |  High 60–79  |  Critical 80–100

Each assessment returns: overall score, severity level, attack category,
model confidence, evidence breakdown, and a recommended response.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional

import numpy as np

from backend.core.config import (
    get_config,
    get_attack_severity,
    ATTACK_SEVERITY,
    RiskWeights,
    RiskThresholds,
)

logger = logging.getLogger(__name__)


# ──────────────────────── Recommended responses ────────────────────────

RECOMMENDED_RESPONSES: Dict[str, Dict[str, Any]] = {
    "low": {
        "action": "monitor",
        "description": "Continue normal monitoring. Log the event for trend analysis.",
        "urgency": "routine",
        "steps": [
            "Log event for historical analysis",
            "No immediate action required",
            "Review during next scheduled audit",
        ],
    },
    "medium": {
        "action": "investigate",
        "description": "Investigate the event. Review related traffic and correlate with other indicators.",
        "urgency": "elevated",
        "steps": [
            "Alert SOC analyst for review",
            "Correlate with recent alerts from same source",
            "Check if source IP has prior incidents",
            "Monitor for escalation",
        ],
    },
    "high": {
        "action": "contain",
        "description": "Initiate containment procedures. Isolate affected hosts if confirmed malicious.",
        "urgency": "high",
        "steps": [
            "Notify SOC team immediately",
            "Isolate affected host(s) from network",
            "Capture forensic evidence (memory dump, disk image)",
            "Block source IP at perimeter firewall",
            "Scan related hosts for indicators of compromise",
        ],
    },
    "critical": {
        "action": "block_and_escalate",
        "description": "Immediate automated blocking recommended. Escalate to incident response team.",
        "urgency": "critical",
        "steps": [
            "Auto-block source IP at firewall (if enabled)",
            "Escalate to incident response team",
            "Isolate affected segment",
            "Initiate full forensic investigation",
            "Notify management / CISO",
            "Preserve all log data for legal review",
        ],
    },
}


# ──────────────────────── Data classes ────────────────────────


@dataclass
class RiskComponents:
    """Individual normalised components feeding into the risk score."""
    model_confidence: float = 0.0   # 0–100
    attack_severity: float = 0.0    # 0–100
    anomaly_score: float = 0.0      # 0–100
    frequency_score: float = 0.0    # 0–100


@dataclass
class RiskAssessment:
    """Full risk assessment result."""
    risk_score: float = 0.0
    severity_level: str = "low"
    attack_category: Optional[str] = None
    model_confidence: float = 0.0
    components: RiskComponents = field(default_factory=RiskComponents)
    weights_used: Dict[str, float] = field(default_factory=dict)
    evidence: Dict[str, Any] = field(default_factory=dict)
    recommended_response: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Serialise to a plain dict."""
        return asdict(self)


# ──────────────────────── Engine ────────────────────────


class RiskEngine:
    """
    Score network incidents on a 0–100 risk scale and return actionable
    recommendations.

    Usage::

        engine = RiskEngine()
        assessment = engine.assess(
            model_confidence=0.92,
            attack_type="ddos",
            anomaly_score=78.0,
            frequency_score=45.0,
        )
        print(assessment.risk_score, assessment.severity_level)
        # 76.1  high

    Parameters
    ----------
    weights : RiskWeights, optional
        Custom weights.  Defaults to global config.
    thresholds : RiskThresholds, optional
        Custom severity boundaries.  Defaults to global config.
    """

    def __init__(
        self,
        weights: Optional[RiskWeights] = None,
        thresholds: Optional[RiskThresholds] = None,
    ) -> None:
        cfg = get_config()
        self._weights: RiskWeights = weights or cfg.risk_weights
        self._thresholds: RiskThresholds = thresholds or cfg.risk_thresholds
        self._validate_weights()
        logger.info(
            "RiskEngine initialised  weights=(mc=%.2f, as=%.2f, an=%.2f, fr=%.2f)",
            self._weights.model_confidence,
            self._weights.attack_severity,
            self._weights.anomaly_score,
            self._weights.frequency_score,
        )

    # ── public API ───────────────────────────────────────

    def assess(
        self,
        model_confidence: float = 0.0,
        attack_type: Optional[str] = None,
        anomaly_score: float = 0.0,
        frequency_score: float = 0.0,
        extra_evidence: Optional[Dict[str, Any]] = None,
    ) -> RiskAssessment:
        """
        Compute a risk assessment for a single incident.

        Parameters
        ----------
        model_confidence : float
            ML model's prediction probability (0.0 – 1.0).  Internally
            normalised to 0–100.
        attack_type : str, optional
            Attack category string (looked up in ``ATTACK_SEVERITY``).
        anomaly_score : float
            Anomaly detection score, already on a 0–100 scale.
        frequency_score : float
            How frequently this source / pattern has been seen (0–100).
        extra_evidence : dict, optional
            Additional evidence to include verbatim in the result.

        Returns
        -------
        RiskAssessment
        """
        # --- normalise components to 0–100 ---
        mc_norm = self._normalise_confidence(model_confidence)
        as_norm = float(np.clip(get_attack_severity(attack_type or ""), 0, 100))
        an_norm = float(np.clip(anomaly_score, 0, 100))
        fr_norm = float(np.clip(frequency_score, 0, 100))

        components = RiskComponents(
            model_confidence=round(mc_norm, 2),
            attack_severity=round(as_norm, 2),
            anomaly_score=round(an_norm, 2),
            frequency_score=round(fr_norm, 2),
        )

        # --- weighted sum ---
        risk_score = (
            self._weights.model_confidence * mc_norm
            + self._weights.attack_severity * as_norm
            + self._weights.anomaly_score * an_norm
            + self._weights.frequency_score * fr_norm
        )
        risk_score = float(np.clip(risk_score, 0, 100))

        # --- severity level ---
        severity = self._score_to_severity(risk_score)

        # --- recommended response ---
        response = self._get_response(severity)

        # --- evidence ---
        evidence: Dict[str, Any] = {
            "components": asdict(components),
            "weights": {
                "model_confidence": self._weights.model_confidence,
                "attack_severity": self._weights.attack_severity,
                "anomaly_score": self._weights.anomaly_score,
                "frequency_score": self._weights.frequency_score,
            },
            "attack_type_raw": attack_type,
            "attack_severity_lookup": as_norm,
        }
        if extra_evidence:
            evidence["extra"] = extra_evidence

        return RiskAssessment(
            risk_score=round(risk_score, 2),
            severity_level=severity,
            attack_category=attack_type,
            model_confidence=round(mc_norm, 2),
            components=components,
            weights_used={
                "model_confidence": self._weights.model_confidence,
                "attack_severity": self._weights.attack_severity,
                "anomaly_score": self._weights.anomaly_score,
                "frequency_score": self._weights.frequency_score,
            },
            evidence=evidence,
            recommended_response=response,
        )

    def assess_batch(
        self,
        incidents: List[Dict[str, Any]],
    ) -> List[RiskAssessment]:
        """
        Score a batch of incidents.

        Each dict in *incidents* should contain the same keyword arguments
        accepted by :meth:`assess` (``model_confidence``, ``attack_type``,
        ``anomaly_score``, ``frequency_score``, ``extra_evidence``).

        Returns
        -------
        list[RiskAssessment]
        """
        return [self.assess(**inc) for inc in incidents]

    # ── severity helpers ─────────────────────────────────

    def _score_to_severity(self, score: float) -> str:
        """Map a 0–100 risk score to a severity label."""
        if score <= self._thresholds.low_max:
            return "low"
        if score <= self._thresholds.medium_max:
            return "medium"
        if score <= self._thresholds.high_max:
            return "high"
        return "critical"

    @staticmethod
    def _get_response(severity: str) -> Dict[str, Any]:
        """Return the recommended response dict for a severity level."""
        return dict(RECOMMENDED_RESPONSES.get(severity, RECOMMENDED_RESPONSES["low"]))

    # ── normalisation ────────────────────────────────────

    @staticmethod
    def _normalise_confidence(confidence: float) -> float:
        """
        Normalise model confidence to 0–100.

        Accepts both 0–1 probability and 0–100 percentage inputs.
        """
        if confidence < 0:
            return 0.0
        if confidence <= 1.0:
            return confidence * 100.0
        return float(np.clip(confidence, 0, 100))

    # ── validation ───────────────────────────────────────

    def _validate_weights(self) -> None:
        """Warn (but do not raise) if weights do not sum to ≈ 1.0."""
        total = (
            self._weights.model_confidence
            + self._weights.attack_severity
            + self._weights.anomaly_score
            + self._weights.frequency_score
        )
        if not (0.99 <= total <= 1.01):
            logger.warning(
                "Risk weights sum to %.4f (expected 1.0). Results may exceed 0–100 range.",
                total,
            )

    # ── convenience class methods ────────────────────────

    @classmethod
    def severity_for(cls, attack_type: str) -> int:
        """Quick lookup of the severity value for an attack type."""
        return get_attack_severity(attack_type)

    @classmethod
    def available_attack_types(cls) -> Dict[str, int]:
        """Return the full ATTACK_SEVERITY mapping."""
        return dict(ATTACK_SEVERITY)
