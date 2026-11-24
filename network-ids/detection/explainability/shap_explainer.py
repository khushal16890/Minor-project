"""
SHAP Explainability Module — compute feature contributions for ML predictions.

Integrates SHAP (SHapley Additive exPlanations) with XGBoost using
``TreeExplainer`` for fast, exact Shapley value computation.

Provides:
- Per-incident feature contributions  (local explanations)
- Global feature importance            (mean |SHAP| values)
- Waterfall / bar chart data           (dict format for dashboard rendering)

Design principles:
- Never invent or fabricate SHAP values — all values come from the SHAP library
- Gracefully handle missing features, unloaded models, and shape mismatches
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np

try:
    import shap
except ImportError:  # pragma: no cover
    shap = None  # type: ignore[assignment]

try:
    import joblib
except ImportError:  # pragma: no cover
    joblib = None  # type: ignore[assignment]

try:
    import xgboost as xgb
except ImportError:  # pragma: no cover
    xgb = None  # type: ignore[assignment]

from backend.core.config import MODELS_DIR, FEATURE_SCHEMA

logger = logging.getLogger(__name__)


class ShapExplainer:
    """
    Compute SHAP feature contributions for XGBoost-based predictions.

    Usage::

        explainer = ShapExplainer()
        explainer.load_model("xgboost_binary")       # loads from models/ dir

        # Per-instance explanation
        contributions = explainer.explain(sample_features)
        # → {"flow_duration": 0.23, "dst_port": -0.11, …}

        # Global importance across a dataset
        importance = explainer.global_importance(X_background)
        # → {"flow_duration": 0.45, …}

    Parameters
    ----------
    model : object, optional
        A pre-loaded XGBoost model (Booster, XGBClassifier, or XGBRegressor).
        If not provided, call :meth:`load_model` before explaining.
    feature_names : list[str], optional
        Ordered feature names matching the model input columns.
        Defaults to ``FEATURE_SCHEMA`` from config.
    """

    def __init__(
        self,
        model: Optional[Any] = None,
        feature_names: Optional[List[str]] = None,
    ) -> None:
        self._model: Optional[Any] = model
        self._feature_names: List[str] = list(feature_names or FEATURE_SCHEMA)
        self._explainer: Optional[shap.TreeExplainer] = None  # type: ignore[union-attr]
        self._is_ready: bool = False

        if self._model is not None:
            self._init_explainer()

    # ── model loading ────────────────────────────────────

    def load_model(self, model_name: str) -> None:
        """
        Load a saved model from the ``models/`` directory via *joblib*.

        Parameters
        ----------
        model_name : str
            Basename of the model file (without ``.joblib`` extension).

        Raises
        ------
        FileNotFoundError
            If the model file does not exist.
        RuntimeError
            If joblib is not installed.
        """
        if joblib is None:
            raise RuntimeError("joblib is required to load models — pip install joblib")

        model_path = Path(MODELS_DIR) / f"{model_name}.joblib"
        if not model_path.exists():
            raise FileNotFoundError(f"Model file not found: {model_path}")

        self._model = joblib.load(model_path)
        logger.info("Loaded model from %s", model_path)

        # Attempt to load feature columns saved alongside the model
        features_path = Path(MODELS_DIR) / f"{model_name}_features.joblib"
        if features_path.exists():
            self._feature_names = joblib.load(features_path)
            logger.info("Loaded %d feature names from %s", len(self._feature_names), features_path)

        self._init_explainer()

    def set_model(self, model: Any, feature_names: Optional[List[str]] = None) -> None:
        """
        Set an already-loaded model (useful when the model lives in memory).

        Parameters
        ----------
        model : object
            XGBoost Booster, XGBClassifier, or XGBRegressor.
        feature_names : list[str], optional
            Ordered feature names; keeps the current list if not provided.
        """
        self._model = model
        if feature_names is not None:
            self._feature_names = list(feature_names)
        self._init_explainer()

    @property
    def is_ready(self) -> bool:
        """``True`` when the explainer is initialised and ready to explain."""
        return self._is_ready

    # ── local explanation ────────────────────────────────

    def explain(
        self,
        features: Union[np.ndarray, List[float], Dict[str, float]],
        top_k: Optional[int] = None,
    ) -> Dict[str, float]:
        """
        Compute per-feature SHAP contributions for a **single** sample.

        Parameters
        ----------
        features : array-like or dict
            Feature values in the same order as ``feature_names``.
            A dict keyed by feature name is also accepted; missing features
            are filled with 0.0.
        top_k : int, optional
            If set, return only the *top_k* features by absolute SHAP value.

        Returns
        -------
        dict[str, float]
            Mapping ``{feature_name: shap_value}``.

        Raises
        ------
        RuntimeError
            If no model / explainer is loaded.
        """
        self._ensure_ready()

        x = self._coerce_input(features)
        shap_values = self._explainer.shap_values(x)  # type: ignore[union-attr]

        # TreeExplainer may return a list (one array per class) for classifiers
        if isinstance(shap_values, list):
            # Use the positive-class (index 1) contributions for binary;
            # for multiclass take the max-abs across classes
            if len(shap_values) == 2:
                sv = np.asarray(shap_values[1]).flatten()
            else:
                sv = np.asarray(shap_values)
                sv = sv[np.argmax(np.abs(sv).sum(axis=-1))].flatten()
        else:
            sv = np.asarray(shap_values).flatten()

        contributions = self._build_contributions(sv, top_k)
        return contributions

    def explain_batch(
        self,
        X: np.ndarray,
        top_k: Optional[int] = None,
    ) -> List[Dict[str, float]]:
        """
        Compute SHAP contributions for every row in *X*.

        Parameters
        ----------
        X : np.ndarray
            2-D array of shape ``(n_samples, n_features)``.
        top_k : int, optional
            If set, keep only the *top_k* features per sample.

        Returns
        -------
        list[dict[str, float]]
        """
        self._ensure_ready()

        X = np.atleast_2d(X)
        shap_values = self._explainer.shap_values(X)  # type: ignore[union-attr]

        if isinstance(shap_values, list):
            sv_array = np.asarray(shap_values[1] if len(shap_values) == 2 else shap_values[0])
        else:
            sv_array = np.asarray(shap_values)

        results: List[Dict[str, float]] = []
        for i in range(sv_array.shape[0]):
            results.append(self._build_contributions(sv_array[i], top_k))
        return results

    # ── global importance ────────────────────────────────

    def global_importance(
        self,
        X: np.ndarray,
        top_k: Optional[int] = None,
    ) -> Dict[str, float]:
        """
        Compute global feature importance as mean |SHAP value| across samples.

        Parameters
        ----------
        X : np.ndarray
            Background dataset, shape ``(n_samples, n_features)``.
        top_k : int, optional
            Return only the *top_k* most important features.

        Returns
        -------
        dict[str, float]
            ``{feature_name: mean_abs_shap_value}`` sorted descending.
        """
        self._ensure_ready()

        X = np.atleast_2d(X)
        shap_values = self._explainer.shap_values(X)  # type: ignore[union-attr]

        if isinstance(shap_values, list):
            sv = np.asarray(shap_values[1] if len(shap_values) == 2 else shap_values[0])
        else:
            sv = np.asarray(shap_values)

        mean_abs = np.mean(np.abs(sv), axis=0)

        n_features = min(len(self._feature_names), len(mean_abs))
        importance: Dict[str, float] = {}
        for idx in range(n_features):
            importance[self._feature_names[idx]] = float(mean_abs[idx])

        # Sort descending
        importance = dict(sorted(importance.items(), key=lambda kv: abs(kv[1]), reverse=True))

        if top_k is not None:
            importance = dict(list(importance.items())[:top_k])

        return importance

    # ── chart data helpers ───────────────────────────────

    def waterfall_data(
        self,
        features: Union[np.ndarray, List[float], Dict[str, float]],
        top_k: int = 15,
    ) -> Dict[str, Any]:
        """
        Return data suitable for rendering a SHAP waterfall chart.

        Parameters
        ----------
        features : array-like or dict
            Single sample.
        top_k : int
            Maximum features to include.

        Returns
        -------
        dict
            Keys: ``features``, ``shap_values``, ``base_value``, ``prediction``.
        """
        self._ensure_ready()

        x = self._coerce_input(features)
        explanation = self._explainer(x)  # type: ignore[misc]

        # Handle multi-output
        if hasattr(explanation, "values") and explanation.values.ndim > 1:
            vals = explanation.values[0]
            base = float(explanation.base_values[0]) if np.ndim(explanation.base_values) > 0 else float(explanation.base_values)
        else:
            vals = np.asarray(explanation.values).flatten()
            base = float(explanation.base_values.flatten()[0]) if np.ndim(explanation.base_values) > 0 else float(explanation.base_values)

        # Pick top_k by magnitude
        abs_vals = np.abs(vals)
        top_indices = np.argsort(abs_vals)[-top_k:][::-1]

        feature_list = []
        value_list = []
        for idx in top_indices:
            fname = self._feature_names[idx] if idx < len(self._feature_names) else f"feature_{idx}"
            feature_list.append(fname)
            value_list.append(float(vals[idx]))

        return {
            "features": feature_list,
            "shap_values": value_list,
            "base_value": base,
            "prediction": base + float(np.sum(vals)),
        }

    def bar_chart_data(
        self,
        X: np.ndarray,
        top_k: int = 20,
    ) -> Dict[str, Any]:
        """
        Return data suitable for rendering a SHAP bar chart of global importance.

        Parameters
        ----------
        X : np.ndarray
            Background dataset.
        top_k : int
            Number of features to include.

        Returns
        -------
        dict
            Keys: ``features``, ``importance_values``.
        """
        importance = self.global_importance(X, top_k=top_k)
        return {
            "features": list(importance.keys()),
            "importance_values": list(importance.values()),
        }

    # ── private helpers ──────────────────────────────────

    def _init_explainer(self) -> None:
        """Create the SHAP TreeExplainer from the loaded model."""
        if shap is None:
            logger.error("SHAP library is not installed — pip install shap")
            self._is_ready = False
            return

        if self._model is None:
            logger.warning("No model set; cannot initialise SHAP explainer.")
            self._is_ready = False
            return

        try:
            self._explainer = shap.TreeExplainer(self._model)
            self._is_ready = True
            logger.info("SHAP TreeExplainer initialised successfully.")
        except Exception:
            logger.exception("Failed to initialise SHAP TreeExplainer")
            self._is_ready = False

    def _ensure_ready(self) -> None:
        """Raise if the explainer is not initialised."""
        if not self._is_ready or self._explainer is None:
            raise RuntimeError(
                "ShapExplainer is not ready. Load a model first with "
                "load_model() or set_model()."
            )

    def _coerce_input(
        self, features: Union[np.ndarray, List[float], Dict[str, float]]
    ) -> np.ndarray:
        """
        Convert input features to a 2-D numpy array aligned with
        ``self._feature_names``.
        """
        if isinstance(features, dict):
            arr = np.zeros((1, len(self._feature_names)), dtype=np.float64)
            for i, name in enumerate(self._feature_names):
                arr[0, i] = features.get(name, 0.0)
            return arr

        arr = np.atleast_2d(np.asarray(features, dtype=np.float64))

        expected = len(self._feature_names)
        actual = arr.shape[1]
        if actual < expected:
            padded = np.zeros((arr.shape[0], expected), dtype=np.float64)
            padded[:, :actual] = arr
            logger.warning(
                "Input has %d features but model expects %d — padded with zeros.",
                actual, expected,
            )
            return padded
        if actual > expected:
            logger.warning(
                "Input has %d features but model expects %d — truncating.",
                actual, expected,
            )
            return arr[:, :expected]
        return arr

    def _build_contributions(
        self, sv: np.ndarray, top_k: Optional[int] = None
    ) -> Dict[str, float]:
        """Map a 1-D SHAP array to ``{feature_name: value}``."""
        n_features = min(len(self._feature_names), len(sv))
        contrib: Dict[str, float] = {}
        for idx in range(n_features):
            contrib[self._feature_names[idx]] = float(sv[idx])

        # Sort by absolute value
        contrib = dict(sorted(contrib.items(), key=lambda kv: abs(kv[1]), reverse=True))

        if top_k is not None:
            contrib = dict(list(contrib.items())[:top_k])

        return contrib
