"""
Model Evaluation Module for the Network Intrusion Detection System.

Computes confusion matrix, classification report, per-class and aggregated
precision / recall / F1, false-positive rate, and inference latency.
Supports cross-dataset evaluation.  Persists results as JSON files and
stores them in the database via ``ModelEvaluation``.
"""

import json
import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

from backend.core.config import MODELS_DIR, EXPERIMENTS_DIR, FEATURE_SCHEMA
from backend.database.models import (
    ModelInfo,
    ModelEvaluation,
    SessionLocal,
    init_db,
)

logger = logging.getLogger(__name__)


class ModelEvaluator:
    """
    Evaluate a trained model against a labelled dataset.

    Parameters
    ----------
    models_dir : Path, optional
        Directory containing ``.joblib`` models and ``_features.json`` files.
    experiments_dir : Path, optional
        Directory to write evaluation JSON reports.
    """

    def __init__(
        self,
        models_dir: Optional[Path] = None,
        experiments_dir: Optional[Path] = None,
    ) -> None:
        self.models_dir = Path(models_dir) if models_dir else MODELS_DIR
        self.experiments_dir = (
            Path(experiments_dir) if experiments_dir else EXPERIMENTS_DIR
        )

    # ── load model helper ────────────────────────────────────────
    def _load_model(self, model_name: str) -> tuple:
        """
        Load a model and its feature columns from disk.

        Returns
        -------
        tuple
            ``(estimator, feature_columns)``
        """
        model_path = self.models_dir / f"{model_name}.joblib"
        features_path = self.models_dir / f"{model_name}_features.json"

        if not model_path.exists():
            raise FileNotFoundError(f"Model not found: {model_path}")

        estimator = joblib.load(model_path)

        feature_columns: List[str] = list(FEATURE_SCHEMA)
        if features_path.exists():
            with open(features_path, "r", encoding="utf-8") as fh:
                feature_columns = json.load(fh)

        return estimator, feature_columns

    # ── main evaluation ──────────────────────────────────────────
    def evaluate(
        self,
        model_name: str,
        X: pd.DataFrame,
        y: pd.Series,
        *,
        task: Optional[str] = None,
        dataset_name: Optional[str] = None,
        save: bool = True,
        store_in_db: bool = True,
    ) -> Dict[str, Any]:
        """
        Evaluate a model on a given dataset.

        Parameters
        ----------
        model_name : str
            Name of the persisted model.
        X : DataFrame
            Feature matrix.
        y : Series
            Ground-truth labels.
        task : str, optional
            ``binary`` or ``multiclass``.  Auto-detected if ``None``.
        dataset_name : str, optional
            Identifier for the evaluation dataset.
        save : bool
            Write JSON report to ``experiments_dir``.
        store_in_db : bool
            Persist results via ``ModelEvaluation`` table.

        Returns
        -------
        dict
            Full evaluation metrics.
        """
        estimator, feature_columns = self._load_model(model_name)

        # Align columns
        X_eval = X.reindex(columns=feature_columns, fill_value=0).copy()
        if "protocol_type" in X_eval.columns and X_eval["protocol_type"].dtype == object:
            proto_map = {"tcp": 6, "udp": 17, "icmp": 1}
            X_eval["protocol_type"] = X_eval["protocol_type"].astype(str).str.lower().map(proto_map).fillna(0)
        for col in X_eval.select_dtypes(include=['object']).columns:
            X_eval[col] = pd.to_numeric(X_eval[col], errors='coerce').fillna(0)
        X_eval.replace([np.inf, -np.inf], np.nan, inplace=True)
        X_eval.fillna(0, inplace=True)

        y_true = y.copy()

        # Auto-detect task
        n_classes = len(y_true.unique())
        if task is None:
            task = "binary" if n_classes <= 2 else "multiclass"

        # Binary-encode or Multiclass-encode if needed
        if task == "binary":
            y_true = y_true.apply(
                lambda lbl: 0
                if str(lbl).strip().lower() in ("benign", "normal", "0")
                else 1
            )
        else:
            if y_true.dtype == object or (len(y_true) > 0 and isinstance(y_true.iloc[0], str)):
                from sklearn.preprocessing import LabelEncoder
                le = LabelEncoder()
                y_true = pd.Series(le.fit_transform(y_true.astype(str)), index=y_true.index)

        # ── predict + latency ────────────────────────────────────
        t0 = time.perf_counter()
        y_pred = estimator.predict(X_eval)
        total_latency = time.perf_counter() - t0
        latency_ms = (total_latency * 1000) / max(len(X_eval), 1)

        # ── metrics ──────────────────────────────────────────────
        avg = "binary" if n_classes <= 2 else "macro"
        cm = confusion_matrix(y_true, y_pred)

        # False positive rate calculation
        fpr = self._compute_fpr(cm, task)

        # Class distribution in test set
        class_dist = y_true.value_counts().to_dict()
        class_dist = {str(k): int(v) for k, v in class_dist.items()}

        report_dict = classification_report(
            y_true, y_pred, output_dict=True, zero_division=0
        )

        metrics: Dict[str, Any] = {
            "model_name": model_name,
            "dataset": dataset_name,
            "task": task,
            "n_samples": int(len(X_eval)),
            "n_classes": n_classes,
            "accuracy": round(float(accuracy_score(y_true, y_pred)), 5),
            "precision_macro": round(
                float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
                5,
            ),
            "recall_macro": round(
                float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
                5,
            ),
            "f1_macro": round(
                float(f1_score(y_true, y_pred, average="macro", zero_division=0)), 5
            ),
            "f1_weighted": round(
                float(f1_score(y_true, y_pred, average="weighted", zero_division=0)), 5
            ),
            "precision_weighted": round(
                float(precision_score(y_true, y_pred, average="weighted", zero_division=0)),
                5,
            ),
            "recall_weighted": round(
                float(recall_score(y_true, y_pred, average="weighted", zero_division=0)),
                5,
            ),
            "false_positive_rate": round(fpr, 5),
            "inference_latency_ms": round(latency_ms, 3),
            "confusion_matrix": cm.tolist(),
            "classification_report": report_dict,
            "class_distribution": class_dist,
        }

        # ── persist ──────────────────────────────────────────────
        if save:
            self._save_json(model_name, dataset_name, metrics)

        if store_in_db:
            self._store_in_db(model_name, metrics)

        logger.info(
            "Evaluation of '%s' on '%s': accuracy=%.4f  F1_macro=%.4f  FPR=%.4f",
            model_name,
            dataset_name or "unknown",
            metrics["accuracy"],
            metrics["f1_macro"],
            metrics["false_positive_rate"],
        )
        return metrics

    # ── cross-dataset evaluation ─────────────────────────────────
    def cross_evaluate(
        self,
        model_name: str,
        datasets: Dict[str, tuple],
        *,
        task: Optional[str] = None,
    ) -> Dict[str, Dict[str, Any]]:
        """
        Evaluate a single model across multiple datasets.

        Parameters
        ----------
        model_name : str
            Name of the trained model.
        datasets : dict
            ``{dataset_name: (X, y)}`` mapping.
        task : str, optional
            ``binary`` or ``multiclass``.

        Returns
        -------
        dict
            ``{dataset_name: metrics_dict}``
        """
        results: Dict[str, Dict[str, Any]] = {}
        for ds_name, (X, y) in datasets.items():
            logger.info("Cross-evaluating '%s' on '%s' ...", model_name, ds_name)
            results[ds_name] = self.evaluate(
                model_name,
                X,
                y,
                task=task,
                dataset_name=ds_name,
                save=True,
                store_in_db=True,
            )
        return results

    # ── false positive rate ──────────────────────────────────────
    @staticmethod
    def _compute_fpr(cm: np.ndarray, task: str) -> float:
        """
        Compute false positive rate from a confusion matrix.

        For binary: FPR = FP / (FP + TN).
        For multiclass: macro-averaged per-class FPR.
        """
        if task == "binary" and cm.shape == (2, 2):
            tn, fp, fn, tp = cm.ravel()
            return float(fp / (fp + tn)) if (fp + tn) > 0 else 0.0

        # Multiclass: one-vs-rest FPR per class, then average
        n_classes = cm.shape[0]
        fprs: List[float] = []
        for i in range(n_classes):
            fp_i = cm[:, i].sum() - cm[i, i]
            tn_i = cm.sum() - cm[i, :].sum() - cm[:, i].sum() + cm[i, i]
            fpr_i = float(fp_i / (fp_i + tn_i)) if (fp_i + tn_i) > 0 else 0.0
            fprs.append(fpr_i)
        return float(np.mean(fprs))

    # ── persistence helpers ──────────────────────────────────────
    def _save_json(
        self,
        model_name: str,
        dataset_name: Optional[str],
        metrics: Dict[str, Any],
    ) -> None:
        """Write evaluation metrics to a JSON file."""
        self.experiments_dir.mkdir(parents=True, exist_ok=True)
        ds_tag = dataset_name or "eval"
        ts = int(time.time())
        filename = f"{model_name}_{ds_tag}_{ts}.json"
        path = self.experiments_dir / filename
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(metrics, fh, indent=2, default=str)
        logger.info("Evaluation JSON saved to %s", path)

    def _store_in_db(self, model_name: str, metrics: Dict[str, Any]) -> None:
        """Persist evaluation results via the ``ModelEvaluation`` table."""
        init_db()
        session = SessionLocal()
        try:
            model_record = (
                session.query(ModelInfo).filter_by(name=model_name).first()
            )
            if model_record is None:
                logger.warning(
                    "Model '%s' not found in DB; skipping evaluation storage.",
                    model_name,
                )
                return

            evaluation = ModelEvaluation(
                model_id=model_record.id,
                dataset=metrics.get("dataset"),
                task=metrics.get("task"),
                accuracy=metrics.get("accuracy"),
                precision_macro=metrics.get("precision_macro"),
                recall_macro=metrics.get("recall_macro"),
                f1_macro=metrics.get("f1_macro"),
                f1_weighted=metrics.get("f1_weighted"),
                false_positive_rate=metrics.get("false_positive_rate"),
                inference_latency_ms=metrics.get("inference_latency_ms"),
                confusion_matrix=metrics.get("confusion_matrix"),
                classification_report=metrics.get("classification_report"),
                class_distribution=metrics.get("class_distribution"),
            )
            session.add(evaluation)
            session.commit()
            logger.info(
                "Evaluation for model '%s' stored in DB (id=%s).",
                model_name,
                evaluation.id,
            )
        except Exception:
            session.rollback()
            logger.exception("Failed to store evaluation in database.")
            raise
        finally:
            session.close()
