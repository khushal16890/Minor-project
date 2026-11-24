"""
Inference Engine for the Network Intrusion Detection System.

Loads trained models from disk, caches them in memory, and provides
single-sample and batch prediction with confidence scores and latency
tracking.
"""

import json
import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import joblib
import numpy as np
import pandas as pd

from backend.core.config import MODELS_DIR, FEATURE_SCHEMA

logger = logging.getLogger(__name__)


class ModelPredictor:
    """
    Inference engine with in-memory model caching.

    Models are loaded lazily on first prediction and kept in a process-level
    cache so subsequent calls avoid disk I/O.

    Parameters
    ----------
    models_dir : Path, optional
        Directory that holds ``.joblib`` model files and ``_features.json``
        companion files.  Defaults to ``MODELS_DIR`` from config.
    """

    # Class-level cache: model_name → (estimator, feature_columns)
    _cache: Dict[str, tuple] = {}

    def __init__(self, models_dir: Optional[Path] = None) -> None:
        self.models_dir = Path(models_dir) if models_dir else MODELS_DIR

    # ── model loading & caching ──────────────────────────────────
    def _load_model(self, model_name: str) -> tuple:
        """
        Load a model and its feature list from disk, caching the result.

        Returns
        -------
        tuple
            ``(estimator, feature_columns)``
        """
        if model_name in self._cache:
            return self._cache[model_name]

        model_path = self.models_dir / f"{model_name}.joblib"
        features_path = self.models_dir / f"{model_name}_features.json"

        if not model_path.exists():
            raise FileNotFoundError(f"Model file not found: {model_path}")

        estimator = joblib.load(model_path)
        logger.info("Loaded model '%s' from %s", model_name, model_path)

        # Feature columns
        feature_columns: List[str] = list(FEATURE_SCHEMA)
        if features_path.exists():
            with open(features_path, "r", encoding="utf-8") as fh:
                feature_columns = json.load(fh)
            logger.info(
                "Loaded %d feature columns for '%s'.",
                len(feature_columns),
                model_name,
            )

        self._cache[model_name] = (estimator, feature_columns)
        return estimator, feature_columns

    def clear_cache(self, model_name: Optional[str] = None) -> None:
        """Remove one or all models from the in-memory cache."""
        if model_name:
            self._cache.pop(model_name, None)
        else:
            self._cache.clear()

    # ── single-sample prediction ─────────────────────────────────
    def predict(
        self,
        model_name: str,
        sample: Union[Dict[str, float], pd.Series],
        *,
        label_map: Optional[Dict[int, str]] = None,
    ) -> Dict[str, Any]:
        """
        Predict a single sample.

        Parameters
        ----------
        model_name : str
            Name of the model to load / use.
        sample : dict or Series
            Feature values keyed by feature name.
        label_map : dict, optional
            Mapping from integer labels to human-readable attack types.
            E.g. ``{0: "benign", 1: "malicious"}``.

        Returns
        -------
        dict
            ``predicted_label``, ``confidence``, ``attack_type``,
            ``inference_latency_ms``, ``probabilities``.
        """
        estimator, feature_columns = self._load_model(model_name)

        # Build feature vector — fill missing with 0
        if isinstance(sample, dict):
            sample = pd.Series(sample)
        row = pd.DataFrame([sample.reindex(feature_columns, fill_value=0).values],
                           columns=feature_columns)
        if "protocol_type" in row.columns and row["protocol_type"].dtype == object:
            proto_map = {"tcp": 6, "udp": 17, "icmp": 1}
            row["protocol_type"] = row["protocol_type"].astype(str).str.lower().map(proto_map).fillna(0)
        for col in row.select_dtypes(include=['object']).columns:
            row[col] = pd.to_numeric(row[col], errors='coerce').fillna(0)
        row.replace([np.inf, -np.inf], np.nan, inplace=True)
        row.fillna(0, inplace=True)

        t0 = time.perf_counter()
        predicted = estimator.predict(row)[0]
        proba = self._predict_proba_safe(estimator, row)
        latency_ms = (time.perf_counter() - t0) * 1000

        confidence = float(np.max(proba)) if proba is not None else None
        probabilities = (
            {str(i): round(float(p), 5) for i, p in enumerate(proba[0])}
            if proba is not None
            else None
        )

        # Resolve attack type
        predicted_label = int(predicted) if isinstance(predicted, (np.integer,)) else predicted
        attack_type = self._resolve_attack_type(predicted_label, label_map)

        return {
            "predicted_label": predicted_label,
            "confidence": round(confidence, 5) if confidence is not None else None,
            "attack_type": attack_type,
            "inference_latency_ms": round(latency_ms, 3),
            "probabilities": probabilities,
        }

    # ── batch prediction ─────────────────────────────────────────
    def predict_batch(
        self,
        model_name: str,
        X: pd.DataFrame,
        *,
        label_map: Optional[Dict[int, str]] = None,
    ) -> Dict[str, Any]:
        """
        Predict a batch of samples.

        Parameters
        ----------
        model_name : str
            Name of the model to load / use.
        X : DataFrame
            Feature matrix.
        label_map : dict, optional
            Mapping from integer labels to attack type strings.

        Returns
        -------
        dict
            ``predictions`` (list of dicts per row), ``total_samples``,
            ``average_latency_ms``, ``total_latency_ms``.
        """
        estimator, feature_columns = self._load_model(model_name)

        # Align columns
        X_aligned = X.reindex(columns=feature_columns, fill_value=0).copy()
        X_aligned.replace([np.inf, -np.inf], np.nan, inplace=True)
        X_aligned.fillna(0, inplace=True)

        t0 = time.perf_counter()
        predicted = estimator.predict(X_aligned)
        proba = self._predict_proba_safe(estimator, X_aligned)
        total_latency_ms = (time.perf_counter() - t0) * 1000

        n = len(X_aligned)
        avg_latency = total_latency_ms / max(n, 1)

        results: List[Dict[str, Any]] = []
        for i in range(n):
            pred_label = (
                int(predicted[i])
                if isinstance(predicted[i], (np.integer,))
                else predicted[i]
            )
            confidence = float(np.max(proba[i])) if proba is not None else None
            attack_type = self._resolve_attack_type(pred_label, label_map)
            results.append(
                {
                    "predicted_label": pred_label,
                    "confidence": round(confidence, 5) if confidence is not None else None,
                    "attack_type": attack_type,
                }
            )

        return {
            "predictions": results,
            "total_samples": n,
            "average_latency_ms": round(avg_latency, 3),
            "total_latency_ms": round(total_latency_ms, 3),
        }

    # ── list available models ────────────────────────────────────
    def list_models(self) -> List[str]:
        """Return names of model files found on disk."""
        return sorted(
            p.stem for p in self.models_dir.glob("*.joblib")
        )

    # ── private helpers ──────────────────────────────────────────
    @staticmethod
    def _predict_proba_safe(estimator: Any, X: pd.DataFrame) -> Optional[np.ndarray]:
        """Get prediction probabilities if the estimator supports them."""
        if hasattr(estimator, "predict_proba"):
            try:
                return estimator.predict_proba(X)
            except Exception:
                return None
        return None

    @staticmethod
    def _resolve_attack_type(
        label: Any, label_map: Optional[Dict[int, str]]
    ) -> Optional[str]:
        """Map a numeric or string label to a human-readable attack type."""
        if label_map and label in label_map:
            return label_map[label]
        # Heuristic for binary
        if label in (0, "0"):
            return "benign"
        if label in (1, "1"):
            return "malicious"
        # Multiclass — label is already the class name
        if isinstance(label, str):
            return label
        return str(label)
