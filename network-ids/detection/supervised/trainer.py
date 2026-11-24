"""
Model Training Engine for the Network Intrusion Detection System.

Supports LogisticRegression, RandomForest, XGBoost, and LightGBM for both
binary (benign / malicious) and multiclass classification tasks.
Applies SMOTE oversampling to the training split only, tracks training time,
and persists models + feature lists to disk and metadata to the DB.
"""

import json
import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
)

import lightgbm as lgb
import xgboost as xgb

from backend.core.config import MODELS_DIR, FEATURE_SCHEMA, EXPERIMENTS_DIR
from backend.database.models import ModelInfo, SessionLocal, init_db

logger = logging.getLogger(__name__)

# ── Supported model types ──────────────────────────────────────────
MODEL_TYPES = {
    "logistic_regression": "LogisticRegression",
    "random_forest": "RandomForest",
    "xgboost": "XGBoost",
    "lightgbm": "LightGBM",
}


class ModelTrainer:
    """
    End-to-end training engine: split → SMOTE → train → evaluate → persist.

    Parameters
    ----------
    model_type : str
        One of ``logistic_regression``, ``random_forest``, ``xgboost``, ``lightgbm``.
    task : str
        ``binary`` or ``multiclass``.
    test_size : float
        Fraction of data reserved for the test split (stratified).
    hyperparameters : dict, optional
        Model-specific keyword arguments forwarded to the estimator constructor.
    random_state : int
        Seed for reproducibility.
    """

    def __init__(
        self,
        model_type: str = "xgboost",
        task: str = "binary",
        test_size: float = 0.2,
        hyperparameters: Optional[Dict[str, Any]] = None,
        random_state: int = 42,
    ) -> None:
        model_type = model_type.lower().strip()
        if model_type not in MODEL_TYPES:
            raise ValueError(
                f"Unsupported model_type '{model_type}'. "
                f"Choose from {list(MODEL_TYPES.keys())}."
            )
        if task not in ("binary", "multiclass"):
            raise ValueError("task must be 'binary' or 'multiclass'.")
        if not (0.05 <= test_size <= 0.5):
            raise ValueError("test_size must be between 0.05 and 0.5.")

        self.model_type = model_type
        self.task = task
        self.test_size = test_size
        self.hyperparameters = hyperparameters or {}
        self.random_state = random_state

        self.model: Any = None
        self.feature_columns: List[str] = []
        self.label_classes_: Optional[np.ndarray] = None

    # ── estimator factory ────────────────────────────────────────
    def _build_estimator(self, n_classes: int) -> Any:
        """Return a fresh, un-fitted estimator instance."""
        params = dict(self.hyperparameters)  # shallow copy

        if self.model_type == "logistic_regression":
            defaults = dict(
                max_iter=1000,
                solver="lbfgs",
                multi_class="auto",
                random_state=self.random_state,
                n_jobs=-1,
            )
            defaults.update(params)
            return LogisticRegression(**defaults)

        if self.model_type == "random_forest":
            defaults = dict(
                n_estimators=200,
                max_depth=None,
                random_state=self.random_state,
                n_jobs=-1,
                class_weight="balanced",
            )
            defaults.update(params)
            return RandomForestClassifier(**defaults)

        if self.model_type == "xgboost":
            objective = (
                "binary:logistic"
                if n_classes == 2
                else "multi:softprob"
            )
            defaults = dict(
                n_estimators=300,
                max_depth=6,
                learning_rate=0.1,
                objective=objective,
                eval_metric="logloss" if n_classes == 2 else "mlogloss",
                use_label_encoder=False,
                random_state=self.random_state,
                n_jobs=-1,
                tree_method="hist",
            )
            if n_classes > 2:
                defaults["num_class"] = n_classes
            defaults.update(params)
            return xgb.XGBClassifier(**defaults)

        if self.model_type == "lightgbm":
            objective = "binary" if n_classes == 2 else "multiclass"
            defaults = dict(
                n_estimators=300,
                max_depth=-1,
                learning_rate=0.1,
                objective=objective,
                random_state=self.random_state,
                n_jobs=-1,
                verbose=-1,
            )
            if n_classes > 2:
                defaults["num_class"] = n_classes
            defaults.update(params)
            return lgb.LGBMClassifier(**defaults)

        raise ValueError(f"Unknown model_type: {self.model_type}")  # pragma: no cover

    # ── public API ───────────────────────────────────────────────
    def train(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        *,
        apply_smote: bool = True,
        smote_strategy: str = "auto",
        model_name: Optional[str] = None,
        dataset_name: Optional[str] = None,
        save: bool = True,
    ) -> Dict[str, Any]:
        """
        Run the full training pipeline.

        Parameters
        ----------
        X : DataFrame
            Feature matrix (should contain only columns in FEATURE_SCHEMA).
        y : Series
            Target labels (string or integer encoded).
        apply_smote : bool
            Whether to apply SMOTE on the training split.
        smote_strategy : str
            SMOTE ``sampling_strategy`` argument.
        model_name : str, optional
            Name used when saving.  Auto-generated if ``None``.
        dataset_name : str, optional
            Dataset identifier stored in metadata.
        save : bool
            Persist the model to disk and register in the database.

        Returns
        -------
        dict
            Training metrics including accuracy, precision, recall, F1,
            classification report, training time, and model path.
        """
        start_time = time.time()

        # ── resolve feature columns ──────────────────────────────
        available = [c for c in FEATURE_SCHEMA if c in X.columns]
        if not available:
            raise ValueError(
                "No FEATURE_SCHEMA columns found in X. "
                "Available columns: " + ", ".join(X.columns[:20].tolist())
            )
        self.feature_columns = available
        X = X[available].copy()

        # ── clean data ───────────────────────────────────────────
        if "protocol_type" in X.columns and X["protocol_type"].dtype == object:
            proto_map = {"tcp": 6, "udp": 17, "icmp": 1}
            X["protocol_type"] = X["protocol_type"].astype(str).str.lower().map(proto_map).fillna(0)
        for col in X.select_dtypes(include=['object']).columns:
            X[col] = pd.to_numeric(X[col], errors='coerce').fillna(0)
        X.replace([np.inf, -np.inf], np.nan, inplace=True)
        X.fillna(0, inplace=True)

        # ── encode labels ────────────────────────────────────────
        y = y.copy()
        if self.task == "binary":
            y = y.apply(
                lambda lbl: 0
                if str(lbl).strip().lower() in ("benign", "normal", "0")
                else 1
            )
        else:
            if y.dtype == object or (len(y) > 0 and isinstance(y.iloc[0], str)):
                from sklearn.preprocessing import LabelEncoder
                le = LabelEncoder()
                y = pd.Series(le.fit_transform(y.astype(str)), index=y.index)
                self.label_encoder_ = le

        self.label_classes_ = np.sort(y.unique())
        n_classes = len(self.label_classes_)
        logger.info(
            "Training %s (%s) — %d samples, %d features, %d classes",
            self.model_type,
            self.task,
            len(X),
            len(self.feature_columns),
            n_classes,
        )

        # ── stratified split ─────────────────────────────────────
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=self.test_size, stratify=y, random_state=self.random_state
        )
        logger.info(
            "Split: train=%d  test=%d  (test_size=%.2f)",
            len(X_train), len(X_test), self.test_size,
        )

        # ── SMOTE on training data only ──────────────────────────
        if apply_smote:
            try:
                # Need at least k_neighbors+1 samples per class for SMOTE
                min_class_count = y_train.value_counts().min()
                k_neighbors = min(5, min_class_count - 1)
                if k_neighbors < 1:
                    logger.warning(
                        "Too few samples for SMOTE (min class has %d). Skipping.",
                        min_class_count,
                    )
                else:
                    smote = SMOTE(
                        sampling_strategy=smote_strategy,
                        k_neighbors=k_neighbors,
                        random_state=self.random_state,
                    )
                    X_train, y_train = smote.fit_resample(X_train, y_train)
                    logger.info("SMOTE applied — training set now %d samples.", len(X_train))
            except Exception as exc:
                logger.warning("SMOTE failed (%s). Training on original distribution.", exc)

        # ── build & fit ──────────────────────────────────────────
        self.model = self._build_estimator(n_classes)
        self.model.fit(X_train, y_train)
        training_time = time.time() - start_time
        logger.info("Model trained in %.2fs.", training_time)

        # ── evaluate on held-out test set ────────────────────────
        y_pred = self.model.predict(X_test)
        avg = "binary" if n_classes == 2 else "macro"

        metrics: Dict[str, Any] = {
            "model_type": self.model_type,
            "task": self.task,
            "n_features": len(self.feature_columns),
            "n_train_samples": int(len(X_train)),
            "n_test_samples": int(len(X_test)),
            "n_classes": n_classes,
            "training_time_seconds": round(training_time, 3),
            "accuracy": round(float(accuracy_score(y_test, y_pred)), 5),
            "precision_macro": round(
                float(precision_score(y_test, y_pred, average="macro", zero_division=0)), 5
            ),
            "recall_macro": round(
                float(recall_score(y_test, y_pred, average="macro", zero_division=0)), 5
            ),
            "f1_macro": round(
                float(f1_score(y_test, y_pred, average="macro", zero_division=0)), 5
            ),
            "f1_weighted": round(
                float(f1_score(y_test, y_pred, average="weighted", zero_division=0)), 5
            ),
            "classification_report": classification_report(
                y_test, y_pred, output_dict=True, zero_division=0
            ),
        }

        # ── persist ──────────────────────────────────────────────
        if save:
            ts = int(time.time())
            if model_name is None:
                model_name = f"{self.model_type}_{self.task}_{ts}"

            model_path = self._save_model(model_name)
            metrics["model_name"] = model_name
            metrics["model_path"] = str(model_path)

            # DB registration
            self._register_in_db(
                model_name=model_name,
                model_path=str(model_path),
                dataset_name=dataset_name,
                n_train_samples=len(X_train),
            )

            # Save experiment JSON
            exp_path = EXPERIMENTS_DIR / f"{model_name}.json"
            exp_path.parent.mkdir(parents=True, exist_ok=True)
            with open(exp_path, "w", encoding="utf-8") as fh:
                json.dump(metrics, fh, indent=2, default=str)
            logger.info("Experiment results saved to %s", exp_path)

        return metrics

    # ── save / load helpers ──────────────────────────────────────
    def _save_model(self, model_name: str) -> Path:
        """Save model and feature columns to ``MODELS_DIR``."""
        MODELS_DIR.mkdir(parents=True, exist_ok=True)

        model_path = MODELS_DIR / f"{model_name}.joblib"
        joblib.dump(self.model, model_path)
        logger.info("Model saved to %s", model_path)

        # Save feature columns alongside the model
        features_path = MODELS_DIR / f"{model_name}_features.json"
        with open(features_path, "w", encoding="utf-8") as fh:
            json.dump(self.feature_columns, fh, indent=2)
        logger.info("Feature columns saved to %s", features_path)

        return model_path

    def _register_in_db(
        self,
        model_name: str,
        model_path: str,
        dataset_name: Optional[str],
        n_train_samples: int,
    ) -> None:
        """Create a ``ModelInfo`` record in the database."""
        init_db()
        session = SessionLocal()
        try:
            existing = session.query(ModelInfo).filter_by(name=model_name).first()
            if existing:
                existing.version += 1  # type: ignore[operator]
                existing.file_path = model_path
                existing.feature_columns = self.feature_columns
                existing.training_samples = n_train_samples
                existing.hyperparameters = self.hyperparameters
            else:
                record = ModelInfo(
                    name=model_name,
                    model_type=self.model_type,
                    task=self.task,
                    dataset=dataset_name,
                    file_path=model_path,
                    feature_columns=self.feature_columns,
                    hyperparameters=self.hyperparameters,
                    training_samples=n_train_samples,
                )
                session.add(record)
            session.commit()
            logger.info("Model '%s' registered in database.", model_name)
        except Exception:
            session.rollback()
            logger.exception("Failed to register model in database.")
            raise
        finally:
            session.close()
