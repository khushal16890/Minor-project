"""
Preprocessing pipeline for the Network Intrusion Detection System.

Provides a single ``PreprocessingPipeline`` class that handles:
* Data cleaning (NaN, inf, duplicates)
* Categorical encoding (protocol_type, etc.)
* Standard-scaling of numeric features
* Save / load of fitted scaler + encoder state via joblib
* Identical preprocessing path for training and inference
"""

import logging
from pathlib import Path
from typing import Optional

import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler

from backend.core.config import FEATURE_SCHEMA, PROCESSED_DATA_DIR

logger = logging.getLogger(__name__)

# Columns that MUST NOT be used as model features
_EXCLUDED_COLUMNS = frozenset({
    "src_ip",
    "dst_ip",
    "timestamp",
    "flow_id",
    "Flow ID",
    " Flow ID",
    "Timestamp",
    " Timestamp",
    "Source IP",
    " Source IP",
    "Destination IP",
    " Destination IP",
})

# Columns in FEATURE_SCHEMA that should be treated as categorical
_CATEGORICAL_FEATURES = ["protocol_type"]


class PreprocessingPipeline:
    """End-to-end preprocessing pipeline.

    Usage — Training::

        pipeline = PreprocessingPipeline()
        X_train, y_train = pipeline.fit_transform(df, label_col="label")
        pipeline.save("my_pipeline")

    Usage — Inference::

        pipeline = PreprocessingPipeline.load("my_pipeline")
        X_test = pipeline.transform(df)

    The pipeline enforces the same column ordering and transformations in both
    ``fit_transform`` and ``transform`` so that training / inference feature
    spaces are always identical.
    """

    def __init__(self) -> None:
        self._scaler: Optional[StandardScaler] = None
        self._label_encoders: dict[str, LabelEncoder] = {}
        self._is_fitted: bool = False
        # Ordered list of numeric feature names after encoding
        self._numeric_features: list[str] = []
        # Complete ordered feature list (after encoding)
        self._feature_order: list[str] = []

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def is_fitted(self) -> bool:
        """Whether the pipeline has been fitted on training data."""
        return self._is_fitted

    @property
    def feature_names(self) -> list[str]:
        """Ordered feature names the pipeline was fitted on."""
        return list(self._feature_order)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fit_transform(
        self,
        df: pd.DataFrame,
        label_col: str = "label",
        *,
        drop_label: bool = True,
    ) -> tuple[np.ndarray, Optional[np.ndarray]]:
        """Clean, encode, fit scaler, and transform training data.

        Parameters
        ----------
        df:
            Raw training DataFrame.  Must contain columns matching
            ``FEATURE_SCHEMA`` (extras and excluded columns are dropped).
        label_col:
            Name of the target column.  If the column is absent and
            ``drop_label`` is ``True``, labels are returned as ``None``.
        drop_label:
            Whether to separate the label column from features.

        Returns
        -------
        (X, y) where X is a 2-D numpy array and y is a 1-D array (or None).
        """
        df = self._clean(df.copy())
        df = self._drop_excluded(df)

        y: Optional[np.ndarray] = None
        if drop_label and label_col in df.columns:
            y = df[label_col].values
            df = df.drop(columns=[label_col])

        # Keep only FEATURE_SCHEMA columns that exist in the data
        available = [c for c in FEATURE_SCHEMA if c in df.columns]
        df = df[available]

        # Encode categoricals
        df = self._fit_encode_categoricals(df)

        # Determine feature order
        self._feature_order = list(df.columns)
        self._numeric_features = [
            c for c in self._feature_order if c not in _CATEGORICAL_FEATURES
        ]

        # Fit scaler on numeric features
        self._scaler = StandardScaler()
        numeric_cols = [c for c in self._numeric_features if c in df.columns]
        if numeric_cols:
            df[numeric_cols] = self._scaler.fit_transform(df[numeric_cols].values)

        self._is_fitted = True
        logger.info(
            "Pipeline fitted on %d samples with %d features",
            len(df),
            len(self._feature_order),
        )

        X = df[self._feature_order].values.astype(np.float32)
        return X, y

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        """Apply the fitted pipeline to new data (inference / test).

        Parameters
        ----------
        df:
            DataFrame with columns matching those seen during ``fit_transform``.
            Missing schema columns are filled with 0.

        Returns
        -------
        2-D numpy array with the same column order as training.

        Raises
        ------
        RuntimeError
            If the pipeline has not been fitted yet.
        """
        if not self._is_fitted:
            raise RuntimeError(
                "Pipeline has not been fitted. Call fit_transform() first or load a saved pipeline."
            )

        df = self._clean(df.copy())
        df = self._drop_excluded(df)

        # Drop label columns if present
        for col in ("label", "Label", " Label", "attack_cat"):
            if col in df.columns:
                df = df.drop(columns=[col])

        # Ensure all expected features exist (fill missing with 0)
        for feat in self._feature_order:
            if feat not in df.columns:
                df[feat] = 0.0

        df = df[self._feature_order]

        # Encode categoricals using fitted encoders
        df = self._transform_encode_categoricals(df)

        # Scale numeric features
        numeric_cols = [c for c in self._numeric_features if c in df.columns]
        if numeric_cols and self._scaler is not None:
            df[numeric_cols] = self._scaler.transform(df[numeric_cols].values)

        return df[self._feature_order].values.astype(np.float32)

    def transform_single(self, feature_dict: dict[str, float]) -> np.ndarray:
        """Transform a single feature dict (e.g. from ``FeatureExtractor``).

        Returns a 2-D array of shape ``(1, n_features)``.
        """
        df = pd.DataFrame([feature_dict])
        return self.transform(df)

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, name: str, directory: Optional[Path] = None) -> Path:
        """Save the fitted pipeline state to disk.

        Parameters
        ----------
        name:
            Base name for the saved files (e.g. ``"v1"``).
        directory:
            Directory to save into.  Defaults to ``PROCESSED_DATA_DIR``.

        Returns
        -------
        Path to the saved ``.joblib`` file.
        """
        if not self._is_fitted:
            raise RuntimeError("Cannot save an unfitted pipeline.")

        save_dir = Path(directory) if directory else PROCESSED_DATA_DIR
        save_dir.mkdir(parents=True, exist_ok=True)
        filepath = save_dir / f"pipeline_{name}.joblib"

        state = {
            "scaler": self._scaler,
            "label_encoders": self._label_encoders,
            "feature_order": self._feature_order,
            "numeric_features": self._numeric_features,
        }
        joblib.dump(state, filepath)
        logger.info("Pipeline saved to %s", filepath)
        return filepath

    @classmethod
    def load(cls, name: str, directory: Optional[Path] = None) -> "PreprocessingPipeline":
        """Load a previously saved pipeline.

        Parameters
        ----------
        name:
            Base name used during ``save()``.
        directory:
            Directory to load from.  Defaults to ``PROCESSED_DATA_DIR``.

        Returns
        -------
        A fitted ``PreprocessingPipeline`` instance ready for ``transform()``.
        """
        load_dir = Path(directory) if directory else PROCESSED_DATA_DIR
        filepath = load_dir / f"pipeline_{name}.joblib"

        if not filepath.exists():
            raise FileNotFoundError(f"No saved pipeline at {filepath}")

        state = joblib.load(filepath)
        pipeline = cls()
        pipeline._scaler = state["scaler"]
        pipeline._label_encoders = state["label_encoders"]
        pipeline._feature_order = state["feature_order"]
        pipeline._numeric_features = state["numeric_features"]
        pipeline._is_fitted = True
        logger.info("Pipeline loaded from %s", filepath)
        return pipeline

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _clean(df: pd.DataFrame) -> pd.DataFrame:
        """Handle NaN, inf values, and duplicates."""
        initial_rows = len(df)

        # Strip whitespace from column names
        df.columns = df.columns.str.strip()

        # Replace inf / -inf with NaN, then fill NaN with 0
        df.replace([np.inf, -np.inf], np.nan, inplace=True)
        nan_counts = df.isna().sum()
        total_nans = int(nan_counts.sum())
        if total_nans > 0:
            logger.info("Filling %d NaN values with 0", total_nans)
        df.fillna(0, inplace=True)

        # Drop duplicates
        before_dedup = len(df)
        df.drop_duplicates(inplace=True)
        dropped = before_dedup - len(df)
        if dropped > 0:
            logger.info(
                "Removed %d duplicate rows (%d → %d)", dropped, before_dedup, len(df)
            )

        return df

    @staticmethod
    def _drop_excluded(df: pd.DataFrame) -> pd.DataFrame:
        """Drop columns that must not be used as features."""
        to_drop = [c for c in df.columns if c in _EXCLUDED_COLUMNS]
        if to_drop:
            logger.debug("Dropping excluded columns: %s", to_drop)
            df = df.drop(columns=to_drop)
        return df

    def _fit_encode_categoricals(self, df: pd.DataFrame) -> pd.DataFrame:
        """Fit label encoders on categorical columns and transform them."""
        for col in _CATEGORICAL_FEATURES:
            if col not in df.columns:
                continue
            le = LabelEncoder()
            df[col] = df[col].astype(str)
            df[col] = le.fit_transform(df[col])
            self._label_encoders[col] = le
            logger.debug(
                "Encoded categorical '%s' with %d classes", col, len(le.classes_)
            )
        return df

    def _transform_encode_categoricals(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply fitted label encoders to categorical columns."""
        for col, le in self._label_encoders.items():
            if col not in df.columns:
                continue
            df[col] = df[col].astype(str)
            # Handle unseen categories: map to -1
            known = set(le.classes_)
            df[col] = df[col].apply(
                lambda val, _known=known, _le=le: (
                    int(_le.transform([val])[0]) if val in _known else -1
                )
            )
        return df
