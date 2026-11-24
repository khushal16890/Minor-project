"""
Anomaly Detection Module using Isolation Forest.

Trains an IsolationForest primarily on benign traffic, predicts anomaly scores
normalised to a 0–100 scale, and keeps anomaly results **separate** from
supervised classification results.  Does NOT auto-classify anomalies as
confirmed attacks.
"""

import json
import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest

from backend.core.config import MODELS_DIR, FEATURE_SCHEMA, get_config
from backend.database.models import AnomalyEvent, SessionLocal, init_db

logger = logging.getLogger(__name__)


class AnomalyDetector:
    """
    Isolation-Forest-based anomaly detector for network traffic.

    Parameters
    ----------
    contamination : float
        Expected fraction of anomalous samples in the training data.
        Used by ``IsolationForest`` and influences the automatic decision
        threshold.
    threshold : float
        Normalised anomaly-score threshold (0–100) above which a sample
        is flagged as anomalous.  Defaults to the value from
        ``DetectionConfig.anomaly_threshold`` mapped to 0–100 scale.
    n_estimators : int
        Number of trees in the forest.
    random_state : int
        Seed for reproducibility.
    models_dir : Path, optional
        Directory to save/load model files.
    """

    _MODEL_PREFIX = "anomaly_iforest"

    def __init__(
        self,
        contamination: float = 0.01,
        threshold: Optional[float] = None,
        n_estimators: int = 200,
        random_state: int = 42,
        models_dir: Optional[Path] = None,
    ) -> None:
        self.contamination = contamination
        self.n_estimators = n_estimators
        self.random_state = random_state
        self.models_dir = Path(models_dir) if models_dir else MODELS_DIR

        # Default threshold from config (map raw score to 0-100 scale)
        if threshold is not None:
            self.threshold = threshold
        else:
            cfg = get_config()
            # config stores the raw IF score threshold (e.g. -0.5);
            # we convert to 0-100 scale: score of -1 → 100, +1 → 0
            self.threshold = self._raw_to_score(cfg.detection.anomaly_threshold)

        self.model: Optional[IsolationForest] = None
        self.feature_columns: List[str] = []

    # ── training ─────────────────────────────────────────────────
    def train(
        self,
        X: pd.DataFrame,
        y: Optional[pd.Series] = None,
        *,
        benign_labels: Optional[List[str]] = None,
        model_name: Optional[str] = None,
        save: bool = True,
    ) -> Dict[str, Any]:
        """
        Train on (primarily) benign traffic.

        Parameters
        ----------
        X : DataFrame
            Feature matrix.
        y : Series, optional
            Labels.  If provided, only benign samples are used for training.
        benign_labels : list[str], optional
            Labels considered benign (default: ``["benign", "normal", "0"]``).
        model_name : str, optional
            Filename stem for the saved model.
        save : bool
            Persist the model to disk.

        Returns
        -------
        dict
            Training summary with n_samples, n_features, threshold, and
            training time.
        """
        start = time.time()

        # Resolve feature columns
        available = [c for c in FEATURE_SCHEMA if c in X.columns]
        if not available:
            raise ValueError(
                "No FEATURE_SCHEMA columns found in X. "
                "Available columns: " + ", ".join(X.columns[:20].tolist())
            )
        self.feature_columns = available
        X_features = X[available].copy()
        if "protocol_type" in X_features.columns and X_features["protocol_type"].dtype == object:
            proto_map = {"tcp": 6, "udp": 17, "icmp": 1}
            X_features["protocol_type"] = X_features["protocol_type"].astype(str).str.lower().map(proto_map).fillna(0)
        for col in X_features.select_dtypes(include=['object']).columns:
            X_features[col] = pd.to_numeric(X_features[col], errors='coerce').fillna(0)
        X_features.replace([np.inf, -np.inf], np.nan, inplace=True)
        X_features.fillna(0, inplace=True)

        # Filter to benign only if labels supplied
        if y is not None:
            if benign_labels is None:
                benign_labels = ["benign", "normal", "0"]
            mask = y.astype(str).str.strip().str.lower().isin(benign_labels)
            X_train = X_features.loc[mask]
            if X_train.empty:
                logger.warning(
                    "No benign samples found with labels %s — using all data.",
                    benign_labels,
                )
                X_train = X_features
            else:
                logger.info(
                    "Filtered %d → %d benign samples for IF training.",
                    len(X_features),
                    len(X_train),
                )
        else:
            X_train = X_features

        # Fit
        self.model = IsolationForest(
            n_estimators=self.n_estimators,
            contamination=self.contamination,
            random_state=self.random_state,
            n_jobs=-1,
        )
        self.model.fit(X_train)
        training_time = time.time() - start
        logger.info(
            "IsolationForest trained on %d samples (%d features) in %.2fs.",
            len(X_train),
            len(available),
            training_time,
        )

        if save:
            if model_name is None:
                model_name = f"{self._MODEL_PREFIX}_{int(time.time())}"
            self._save(model_name)

        return {
            "model_name": model_name,
            "n_train_samples": int(len(X_train)),
            "n_features": len(available),
            "contamination": self.contamination,
            "threshold": self.threshold,
            "training_time_seconds": round(training_time, 3),
        }

    # ── prediction ───────────────────────────────────────────────
    def predict(
        self,
        X: pd.DataFrame,
        *,
        store_in_db: bool = False,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> pd.DataFrame:
        """
        Predict anomaly scores for each sample.

        Parameters
        ----------
        X : DataFrame
            Feature matrix.
        store_in_db : bool
            Persist flagged anomalies to the ``AnomalyEvent`` table.
        metadata : dict, optional
            Extra fields (``src_ip``, ``dst_ip``, ``protocol``, ``flow_id``)
            to attach when storing in DB.  Keys map to column names or
            constant values.

        Returns
        -------
        DataFrame
            Original index with columns:
            ``anomaly_score`` (0–100), ``is_anomaly`` (bool),
            ``raw_score`` (original IF decision_function value).
        """
        if self.model is None:
            raise RuntimeError("Model not trained or loaded. Call train() or load() first.")

        X_eval = X.reindex(columns=self.feature_columns, fill_value=0).copy()
        if "protocol_type" in X_eval.columns and X_eval["protocol_type"].dtype == object:
            proto_map = {"tcp": 6, "udp": 17, "icmp": 1}
            X_eval["protocol_type"] = X_eval["protocol_type"].astype(str).str.lower().map(proto_map).fillna(0)
        for col in X_eval.select_dtypes(include=['object']).columns:
            X_eval[col] = pd.to_numeric(X_eval[col], errors='coerce').fillna(0)
        X_eval.replace([np.inf, -np.inf], np.nan, inplace=True)
        X_eval.fillna(0, inplace=True)

        raw_scores = self.model.decision_function(X_eval)
        normalised = np.array([self._raw_to_score(s) for s in raw_scores])
        is_anomaly = normalised >= self.threshold

        result = pd.DataFrame(
            {
                "anomaly_score": np.round(normalised, 2),
                "is_anomaly": is_anomaly,
                "raw_score": np.round(raw_scores, 6),
            },
            index=X.index,
        )

        n_anomalies = int(is_anomaly.sum())
        logger.info(
            "Anomaly detection: %d / %d samples flagged (threshold=%.1f).",
            n_anomalies,
            len(X_eval),
            self.threshold,
        )

        if store_in_db and n_anomalies > 0:
            self._store_anomalies(result[result["is_anomaly"]], X, metadata)

        return result

    def predict_single(
        self,
        sample: Dict[str, float],
    ) -> Dict[str, Any]:
        """
        Score a single sample.

        Returns
        -------
        dict
            ``anomaly_score`` (0–100), ``is_anomaly``, ``raw_score``.
        """
        df = pd.DataFrame([sample])
        res = self.predict(df)
        row = res.iloc[0]
        return {
            "anomaly_score": float(row["anomaly_score"]),
            "is_anomaly": bool(row["is_anomaly"]),
            "raw_score": float(row["raw_score"]),
        }

    # ── save / load ──────────────────────────────────────────────
    def _save(self, model_name: str) -> Path:
        """Persist model and feature columns to disk."""
        self.models_dir.mkdir(parents=True, exist_ok=True)

        model_path = self.models_dir / f"{model_name}.joblib"
        joblib.dump(self.model, model_path)

        meta_path = self.models_dir / f"{model_name}_meta.json"
        meta = {
            "feature_columns": self.feature_columns,
            "contamination": self.contamination,
            "threshold": self.threshold,
            "n_estimators": self.n_estimators,
        }
        with open(meta_path, "w", encoding="utf-8") as fh:
            json.dump(meta, fh, indent=2)

        logger.info("Anomaly model saved to %s", model_path)
        return model_path

    def load(self, model_name: str) -> None:
        """
        Load a previously saved anomaly model from disk.

        Parameters
        ----------
        model_name : str
            Filename stem of the model (without ``.joblib``).
        """
        model_path = self.models_dir / f"{model_name}.joblib"
        meta_path = self.models_dir / f"{model_name}_meta.json"

        if not model_path.exists():
            raise FileNotFoundError(f"Anomaly model not found: {model_path}")

        self.model = joblib.load(model_path)
        logger.info("Anomaly model loaded from %s", model_path)

        if meta_path.exists():
            with open(meta_path, "r", encoding="utf-8") as fh:
                meta = json.load(fh)
            self.feature_columns = meta.get("feature_columns", list(FEATURE_SCHEMA))
            self.contamination = meta.get("contamination", self.contamination)
            self.threshold = meta.get("threshold", self.threshold)
            self.n_estimators = meta.get("n_estimators", self.n_estimators)
        else:
            self.feature_columns = list(FEATURE_SCHEMA)

    # ── score normalisation ──────────────────────────────────────
    @staticmethod
    def _raw_to_score(raw: float) -> float:
        """
        Convert an Isolation Forest ``decision_function`` value to a 0–100
        anomaly score.

        The IF decision function returns values roughly in [-1, 1] where
        negative = more anomalous.  We map:
          -1 → 100  (most anomalous)
           0 →  50
          +1 →   0  (most normal)
        """
        score = max(0.0, min(100.0, (1.0 - raw) * 50.0))
        return round(score, 2)

    # ── DB persistence ───────────────────────────────────────────
    def _store_anomalies(
        self,
        anomaly_df: pd.DataFrame,
        X_original: pd.DataFrame,
        metadata: Optional[Dict[str, Any]],
    ) -> None:
        """Write flagged anomalies to the ``AnomalyEvent`` table."""
        init_db()
        session = SessionLocal()
        try:
            for idx in anomaly_df.index:
                row = anomaly_df.loc[idx]

                # Build a compact features summary (top 5 deviation cols)
                features_summary = None
                if idx in X_original.index:
                    orig = X_original.loc[idx]
                    top_feats = (
                        orig.reindex(self.feature_columns, fill_value=0)
                        .abs()
                        .nlargest(5)
                    )
                    features_summary = {
                        str(k): round(float(v), 4) for k, v in top_feats.items()
                    }

                event = AnomalyEvent(
                    anomaly_score=float(row["anomaly_score"]),
                    threshold=self.threshold,
                    is_anomaly=True,
                    src_ip=(metadata or {}).get("src_ip"),
                    dst_ip=(metadata or {}).get("dst_ip"),
                    protocol=(metadata or {}).get("protocol"),
                    flow_id=(metadata or {}).get("flow_id"),
                    features_summary=features_summary,
                )
                session.add(event)

            session.commit()
            logger.info(
                "Stored %d anomaly events in database.", len(anomaly_df)
            )
        except Exception:
            session.rollback()
            logger.exception("Failed to store anomaly events.")
            raise
        finally:
            session.close()
