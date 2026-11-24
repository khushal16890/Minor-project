"""
Hybrid Detection Engine — aggregates signature, ML, and anomaly detection outputs.

Combines evidence from Suricata signature matching, supervised ML classification,
and Isolation Forest anomaly detection using weighted scoring to produce unified
DetectionIncident records.  Weights are configurable via DetectionConfig.

Design principles:
- Weighted scoring (NOT simple majority voting)
- Uncertainty is kept visible — no silent promotion of low-confidence signals
- Each detection method contributes independently; the hybrid score reflects
  the combined strength of all available evidence
"""

from __future__ import annotations

import datetime
import logging
import uuid
from dataclasses import dataclass, field, asdict
from enum import Enum
from typing import Any, Dict, List, Optional

import numpy as np

from backend.core.config import get_config, DetectionConfig
from backend.database.models import (
    DetectionIncident,
    NetworkFlow,
    AnomalyEvent,
    SuricataAlert,
    SessionLocal,
    init_db,
)

logger = logging.getLogger(__name__)


# ──────────────────────── Data classes ────────────────────────


class DetectionCategory(str, Enum):
    """Classification of the combined detection outcome."""
    CONFIRMED_SIGNATURE = "confirmed_signature_match"
    ML_PREDICTED = "ml_predicted_attack"
    ANOMALOUS_BEHAVIOUR = "anomalous_behaviour"
    CORRELATED_HIGH_CONFIDENCE = "correlated_high_confidence"
    BENIGN = "benign"


@dataclass
class SignatureEvidence:
    """Evidence from a Suricata signature match."""
    signature_id: Optional[int] = None
    signature_msg: Optional[str] = None
    category: Optional[str] = None
    severity: Optional[int] = None
    matched: bool = False


@dataclass
class MLEvidence:
    """Evidence from ML classification."""
    predicted_label: Optional[str] = None
    attack_type: Optional[str] = None
    confidence: float = 0.0
    model_name: Optional[str] = None


@dataclass
class AnomalyEvidence:
    """Evidence from Isolation Forest anomaly detection."""
    anomaly_score: float = 0.0  # normalised 0–100
    is_anomaly: bool = False
    threshold: Optional[float] = None


@dataclass
class HybridResult:
    """Complete result from the hybrid detection engine."""
    hybrid_score: float = 0.0
    detection_category: str = DetectionCategory.BENIGN.value
    detection_methods: List[str] = field(default_factory=list)
    predicted_attack: Optional[str] = None
    model_confidence: float = 0.0
    anomaly_score: float = 0.0
    signature: SignatureEvidence = field(default_factory=SignatureEvidence)
    ml: MLEvidence = field(default_factory=MLEvidence)
    anomaly: AnomalyEvidence = field(default_factory=AnomalyEvidence)
    evidence: Dict[str, Any] = field(default_factory=dict)
    uncertainty_notes: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Serialise to a plain dict suitable for JSON / DB storage."""
        return asdict(self)


# ──────────────────────── Engine ────────────────────────


class HybridDetectionEngine:
    """
    Aggregates detection outputs from signature, ML, and anomaly engines
    using configurable weighted scoring.

    Default weights (from DetectionConfig):
        * Signature match : 0.90  (highest confidence)
        * ML prediction   : 0.70
        * Anomaly detection: 0.50

    Usage::

        engine = HybridDetectionEngine()
        result = engine.analyse(
            signature=SignatureEvidence(matched=True, signature_id=2001, ...),
            ml=MLEvidence(predicted_label="malicious", confidence=0.92, ...),
            anomaly=AnomalyEvidence(anomaly_score=78.0, is_anomaly=True),
            flow_metadata={"src_ip": "10.0.0.5", ...},
        )
        # result is a HybridResult with hybrid_score, detection_category, etc.
    """

    def __init__(self, config: Optional[DetectionConfig] = None) -> None:
        self._cfg: DetectionConfig = config or get_config().detection
        self._signature_weight: float = self._cfg.signature_weight
        self._ml_weight: float = self._cfg.ml_weight
        self._anomaly_weight: float = self._cfg.anomaly_weight
        logger.info(
            "HybridDetectionEngine initialised  weights=(sig=%.2f, ml=%.2f, anom=%.2f)",
            self._signature_weight, self._ml_weight, self._anomaly_weight,
        )

    # ── public API ───────────────────────────────────────

    def analyse(
        self,
        signature: Optional[SignatureEvidence] = None,
        ml: Optional[MLEvidence] = None,
        anomaly: Optional[AnomalyEvidence] = None,
        flow_metadata: Optional[Dict[str, Any]] = None,
    ) -> HybridResult:
        """
        Combine all available detection outputs into a single *HybridResult*.

        Parameters
        ----------
        signature : SignatureEvidence, optional
            Suricata alert evidence (may be ``None`` if no alert fired).
        ml : MLEvidence, optional
            Supervised ML prediction evidence.
        anomaly : AnomalyEvidence, optional
            Isolation Forest anomaly evidence.
        flow_metadata : dict, optional
            Extra metadata to attach (src_ip, dst_ip, ports, protocol, …).

        Returns
        -------
        HybridResult
        """
        sig = signature or SignatureEvidence()
        ml_ev = ml or MLEvidence()
        anom = anomaly or AnomalyEvidence()
        meta = flow_metadata or {}

        # --- individual component scores (0–1 range) ---
        sig_score = self._score_signature(sig)
        ml_score = self._score_ml(ml_ev)
        anom_score = self._score_anomaly(anom)

        # --- weighted combination ---
        total_weight = 0.0
        weighted_sum = 0.0

        active_methods: List[str] = []
        uncertainty_notes: List[str] = []

        if sig.matched:
            weighted_sum += sig_score * self._signature_weight
            total_weight += self._signature_weight
            active_methods.append("signature")
        else:
            uncertainty_notes.append("No signature match — relying on ML/anomaly only.")

        if ml_ev.predicted_label and ml_ev.confidence > 0:
            weighted_sum += ml_score * self._ml_weight
            total_weight += self._ml_weight
            active_methods.append("ml")
            if ml_ev.confidence < self._cfg.ml_confidence_threshold:
                uncertainty_notes.append(
                    f"ML confidence ({ml_ev.confidence:.2f}) below threshold "
                    f"({self._cfg.ml_confidence_threshold:.2f})."
                )
        else:
            uncertainty_notes.append("No ML prediction available.")

        if anom.anomaly_score > 0 or anom.is_anomaly:
            weighted_sum += anom_score * self._anomaly_weight
            total_weight += self._anomaly_weight
            active_methods.append("anomaly")
        else:
            uncertainty_notes.append("No anomaly signal detected.")

        hybrid_score = (weighted_sum / total_weight * 100.0) if total_weight > 0 else 0.0
        hybrid_score = float(np.clip(hybrid_score, 0.0, 100.0))

        # --- determine detection category ---
        category = self._categorise(sig, ml_ev, anom, hybrid_score)

        # --- predicted attack label ---
        predicted_attack = self._resolve_attack_label(sig, ml_ev)

        # --- build evidence payload ---
        evidence = {
            "signature": asdict(sig),
            "ml": asdict(ml_ev),
            "anomaly": asdict(anom),
            "hybrid_score": round(hybrid_score, 2),
            "active_methods": active_methods,
            "weights_used": {
                "signature": self._signature_weight,
                "ml": self._ml_weight,
                "anomaly": self._anomaly_weight,
            },
            **meta,
        }

        return HybridResult(
            hybrid_score=round(hybrid_score, 2),
            detection_category=category.value,
            detection_methods=active_methods,
            predicted_attack=predicted_attack,
            model_confidence=float(ml_ev.confidence),
            anomaly_score=float(anom.anomaly_score),
            signature=sig,
            ml=ml_ev,
            anomaly=anom,
            evidence=evidence,
            uncertainty_notes=uncertainty_notes,
        )

    def create_incident(
        self,
        result: HybridResult,
        flow_id: Optional[str] = None,
        persist: bool = True,
    ) -> DetectionIncident:
        """
        Build (and optionally persist) a ``DetectionIncident`` DB record from
        a *HybridResult*.

        Parameters
        ----------
        result : HybridResult
            Output of :py:meth:`analyse`.
        flow_id : str, optional
            Foreign key to a ``NetworkFlow`` row.
        persist : bool
            If ``True`` (default), write the record to the database.

        Returns
        -------
        DetectionIncident
        """
        detection_method = (
            "hybrid" if len(result.detection_methods) > 1
            else (result.detection_methods[0] if result.detection_methods else "unknown")
        )

        incident = DetectionIncident(
            id=str(uuid.uuid4()),
            flow_id=flow_id,
            timestamp=datetime.datetime.now(datetime.timezone.utc),
            detection_method=detection_method,
            predicted_label=result.predicted_attack or "unknown",
            attack_type=result.predicted_attack,
            model_confidence=result.model_confidence,
            anomaly_score=result.anomaly_score,
            risk_score=result.hybrid_score,
            severity=self._hybrid_score_to_severity(result.hybrid_score),
            signature_id=str(result.signature.signature_id) if result.signature.signature_id else None,
            signature_msg=result.signature.signature_msg,
            src_ip=result.evidence.get("src_ip"),
            dst_ip=result.evidence.get("dst_ip"),
            src_port=result.evidence.get("src_port"),
            dst_port=result.evidence.get("dst_port"),
            protocol=result.evidence.get("protocol"),
            evidence=result.to_dict(),
            status="open",
        )

        if persist:
            self._save_incident(incident)

        return incident

    def analyse_and_record(
        self,
        signature: Optional[SignatureEvidence] = None,
        ml: Optional[MLEvidence] = None,
        anomaly: Optional[AnomalyEvidence] = None,
        flow_metadata: Optional[Dict[str, Any]] = None,
        flow_id: Optional[str] = None,
    ) -> tuple[HybridResult, DetectionIncident]:
        """Convenience: analyse + create_incident in one call."""
        result = self.analyse(signature=signature, ml=ml, anomaly=anomaly, flow_metadata=flow_metadata)
        incident = self.create_incident(result, flow_id=flow_id)
        return result, incident

    # ── private helpers ──────────────────────────────────

    @staticmethod
    def _score_signature(sig: SignatureEvidence) -> float:
        """Score a signature match on a 0–1 scale."""
        if not sig.matched:
            return 0.0
        # Suricata severity: 1 = highest, 4 = lowest
        if sig.severity is not None and 1 <= sig.severity <= 4:
            return 1.0 - (sig.severity - 1) * 0.15  # 1.0, 0.85, 0.70, 0.55
        return 0.90  # default high confidence for a matched signature

    @staticmethod
    def _score_ml(ml_ev: MLEvidence) -> float:
        """Score an ML prediction on a 0–1 scale (== confidence)."""
        if ml_ev.predicted_label is None:
            return 0.0
        if ml_ev.predicted_label.lower() in ("benign", "normal"):
            return 0.0
        return float(np.clip(ml_ev.confidence, 0.0, 1.0))

    @staticmethod
    def _score_anomaly(anom: AnomalyEvidence) -> float:
        """Score anomaly evidence on a 0–1 scale (normalised score / 100)."""
        if not anom.is_anomaly:
            return 0.0
        return float(np.clip(anom.anomaly_score / 100.0, 0.0, 1.0))

    @staticmethod
    def _categorise(
        sig: SignatureEvidence,
        ml_ev: MLEvidence,
        anom: AnomalyEvidence,
        hybrid_score: float,
    ) -> DetectionCategory:
        """Determine the detection category from the available evidence."""
        has_sig = sig.matched
        has_ml = (
            ml_ev.predicted_label is not None
            and ml_ev.predicted_label.lower() not in ("benign", "normal")
            and ml_ev.confidence > 0
        )
        has_anom = anom.is_anomaly

        if has_sig and has_ml and hybrid_score >= 70:
            return DetectionCategory.CORRELATED_HIGH_CONFIDENCE
        if has_sig:
            return DetectionCategory.CONFIRMED_SIGNATURE
        if has_ml:
            return DetectionCategory.ML_PREDICTED
        if has_anom:
            return DetectionCategory.ANOMALOUS_BEHAVIOUR
        return DetectionCategory.BENIGN

    @staticmethod
    def _resolve_attack_label(sig: SignatureEvidence, ml_ev: MLEvidence) -> Optional[str]:
        """Pick the best attack label from available evidence.

        Signature category is preferred when available because it comes from
        curated rule-sets; ML prediction is the fallback.
        """
        if sig.matched and sig.category:
            return sig.category
        if ml_ev.predicted_label and ml_ev.predicted_label.lower() not in ("benign", "normal"):
            return ml_ev.attack_type or ml_ev.predicted_label
        return None

    @staticmethod
    def _hybrid_score_to_severity(score: float) -> str:
        """Map hybrid score to a human-readable severity label."""
        if score >= 80:
            return "critical"
        if score >= 60:
            return "high"
        if score >= 30:
            return "medium"
        return "low"

    @staticmethod
    def _save_incident(incident: DetectionIncident) -> None:
        """Persist a DetectionIncident to the database."""
        try:
            session = SessionLocal()
            try:
                session.add(incident)
                session.commit()
                logger.info("Saved DetectionIncident %s", incident.id)
            except Exception:
                session.rollback()
                logger.exception("Failed to save DetectionIncident %s", incident.id)
                raise
            finally:
                session.close()
        except Exception:
            logger.exception("Database error while saving incident")
