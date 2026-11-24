"""
Dataset loader for the Network Intrusion Detection System.

Handles importing, column-mapping, validation, and label encoding for:
* **CIC-IDS2017** — Canadian Institute for Cybersecurity IDS 2017 dataset
* **UNSW-NB15** — University of New South Wales NB15 dataset
* Any CSV whose columns can be mapped to ``FEATURE_SCHEMA``
"""

import logging
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd

from backend.core.config import FEATURE_SCHEMA, RAW_DATA_DIR
from backend.schemas.schemas import DatasetValidationResponse

logger = logging.getLogger(__name__)

# ═══════════════════════════════════════════════════════════════════════
# Column mapping dictionaries
# ═══════════════════════════════════════════════════════════════════════

# CIC-IDS2017 columns → FEATURE_SCHEMA names
# Note: CIC-IDS2017 columns have leading spaces (e.g. " Flow Duration")
CIC_IDS2017_COLUMN_MAP: dict[str, str] = {
    "Flow Duration":        "flow_duration",
    "Protocol":             "protocol_type",
    "Source Port":          "src_port",
    "Destination Port":     "dst_port",
    "Total Fwd Packets":    "fwd_packets",
    "Total Backward Packets": "bwd_packets",
    "Total Length of Fwd Packets":   "total_fwd_bytes",
    "Total Length of Bwd Packets":   "total_bwd_bytes",
    "Fwd Packet Length Max":   "fwd_pkt_len_max",
    "Fwd Packet Length Min":   "fwd_pkt_len_min",
    "Fwd Packet Length Mean":  "fwd_pkt_len_mean",
    "Fwd Packet Length Std":   "fwd_pkt_len_std",
    "Bwd Packet Length Max":   "bwd_pkt_len_max",
    "Bwd Packet Length Min":   "bwd_pkt_len_min",
    "Bwd Packet Length Mean":  "bwd_pkt_len_mean",
    "Bwd Packet Length Std":   "bwd_pkt_len_std",
    "Flow Bytes/s":            "flow_bytes_per_sec",
    "Flow Packets/s":          "flow_pkts_per_sec",
    "Flow IAT Mean":           "flow_iat_mean",
    "Flow IAT Std":            "flow_iat_std",
    "Flow IAT Max":            "flow_iat_max",
    "Flow IAT Min":            "flow_iat_min",
    "Fwd IAT Mean":            "fwd_iat_mean",
    "Fwd IAT Std":             "fwd_iat_std",
    "Fwd IAT Max":             "fwd_iat_max",
    "Fwd IAT Min":             "fwd_iat_min",
    "Bwd IAT Mean":            "bwd_iat_mean",
    "Bwd IAT Std":             "bwd_iat_std",
    "Bwd IAT Max":             "bwd_iat_max",
    "Bwd IAT Min":             "bwd_iat_min",
    "Fwd PSH Flags":           "fwd_psh_flags",
    "Bwd PSH Flags":           "bwd_psh_flags",
    "Fwd URG Flags":           "fwd_urg_flags",
    "Bwd URG Flags":           "bwd_urg_flags",
    "FIN Flag Count":          "fin_flag_cnt",
    "SYN Flag Count":          "syn_flag_cnt",
    "RST Flag Count":          "rst_flag_cnt",
    "PSH Flag Count":          "psh_flag_cnt",
    "ACK Flag Count":          "ack_flag_cnt",
    "URG Flag Count":          "urg_flag_cnt",
    "ECE Flag Count":          "ece_flag_cnt",
    "CWE Flag Count":          "cwe_flag_cnt",
    "Average Packet Size":     "avg_pkt_size",
    "Avg Fwd Segment Size":    "fwd_seg_size_avg",
    "Avg Bwd Segment Size":    "bwd_seg_size_avg",
    "Subflow Fwd Packets":     "subflow_fwd_pkts",
    "Subflow Bwd Packets":     "subflow_bwd_pkts",
    "Subflow Fwd Bytes":       "subflow_fwd_bytes",
    "Subflow Bwd Bytes":       "subflow_bwd_bytes",
    "Init_Win_bytes_forward":  "init_win_bytes_fwd",
    "Init_Win_bytes_backward": "init_win_bytes_bwd",
    "act_data_pkt_fwd":        "act_data_pkt_fwd",
    "min_seg_size_forward":    "min_seg_size_fwd",
    "Active Mean":             "active_mean",
    "Active Std":              "active_std",
    "Active Max":              "active_max",
    "Active Min":              "active_min",
    "Idle Mean":               "idle_mean",
    "Idle Std":                "idle_std",
    "Idle Max":                "idle_max",
    "Idle Min":                "idle_min",
    # Packet Length stats (some CIC-IDS2017 versions)
    "Packet Length Mean":      "pkt_len_mean",
    "Packet Length Std":       "pkt_len_std",
    "Packet Length Variance":  "pkt_len_var",
    # Label
    "Label":                   "label",
}

# UNSW-NB15 columns → FEATURE_SCHEMA names
UNSW_NB15_COLUMN_MAP: dict[str, str] = {
    "dur":          "flow_duration",
    "proto":        "protocol_type",
    "sport":        "src_port",
    "dsport":       "dst_port",
    "spkts":        "fwd_packets",
    "dpkts":        "bwd_packets",
    "sbytes":       "total_fwd_bytes",
    "dbytes":       "total_bwd_bytes",
    "sttl":         "init_win_bytes_fwd",   # approximate mapping
    "dttl":         "init_win_bytes_bwd",   # approximate mapping
    "smean":        "fwd_pkt_len_mean",
    "dmean":        "bwd_pkt_len_mean",
    "sinpkt":       "fwd_iat_mean",
    "dinpkt":       "bwd_iat_mean",
    "sjit":         "fwd_iat_std",
    "djit":         "bwd_iat_std",
    "swin":         "init_win_bytes_fwd",
    "dwin":         "init_win_bytes_bwd",
    "ct_srv_src":   "conn_frequency",
    "attack_cat":   "attack_category",      # will be used for multiclass label
    "label":        "label",
    # Flags
    "is_sm_ips_ports": "subflow_fwd_pkts",  # approximate mapping
    "ct_state_ttl":    "active_mean",       # approximate mapping
    "ct_flw_http_mthd": "act_data_pkt_fwd", # approximate mapping
    "is_ftp_login":    "fwd_psh_flags",     # approximate mapping
    "ct_ftp_cmd":      "bwd_psh_flags",     # approximate mapping
    "ct_srv_dst":      "subflow_bwd_pkts",  # approximate mapping
    "ct_dst_ltm":      "idle_mean",         # approximate mapping
    "ct_src_ltm":      "active_min",        # approximate mapping
    "ct_src_dport_ltm": "flow_iat_min",     # approximate mapping
    "ct_dst_sport_ltm": "bwd_iat_min",      # approximate mapping
    "ct_dst_src_ltm":  "flow_iat_max",      # approximate mapping
}


# ═══════════════════════════════════════════════════════════════════════
# Label encoding helpers
# ═══════════════════════════════════════════════════════════════════════

# CIC-IDS2017 attack type → binary label
_CIC_BINARY_MAP = {
    "BENIGN": 0,
}
# Anything not "BENIGN" is mapped to 1

# CIC-IDS2017 attack type → multiclass integer
_CIC_MULTI_MAP = {
    "BENIGN": 0,
    "DoS Hulk": 1,
    "PortScan": 2,
    "DDoS": 3,
    "DoS GoldenEye": 4,
    "FTP-Patator": 5,
    "SSH-Patator": 6,
    "DoS slowloris": 7,
    "DoS Slowhttptest": 8,
    "Bot": 9,
    "Web Attack – Brute Force": 10,
    "Web Attack – XSS": 11,
    "Infiltration": 12,
    "Web Attack – Sql Injection": 13,
    "Heartbleed": 14,
}

# UNSW-NB15 attack categories → multiclass integer
_UNSW_MULTI_MAP = {
    "Normal": 0,
    "Generic": 1,
    "Exploits": 2,
    "Fuzzers": 3,
    "DoS": 4,
    "Reconnaissance": 5,
    "Analysis": 6,
    "Backdoor": 7,
    "Shellcode": 8,
    "Worms": 9,
}


class DatasetLoader:
    """Load, validate, and map benchmark datasets to the canonical feature schema.

    Supported datasets:
    * ``cic-ids2017`` — CIC-IDS2017 CSVs
    * ``unsw-nb15`` — UNSW-NB15 CSVs
    * ``generic`` — Any CSV with columns matching ``FEATURE_SCHEMA``
    """

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def load(
        self,
        file_path: str | Path,
        dataset_type: str = "auto",
        *,
        task: str = "binary",
        sample_frac: Optional[float] = None,
        random_state: int = 42,
    ) -> tuple[pd.DataFrame, "DatasetValidationResponse"]:
        """Load a dataset CSV and return a mapped DataFrame + validation info.

        Parameters
        ----------
        file_path:
            Path to the CSV file.
        dataset_type:
            One of ``"cic-ids2017"``, ``"unsw-nb15"``, ``"generic"``, or
            ``"auto"`` (auto-detect from column names).
        task:
            ``"binary"`` or ``"multiclass"`` label encoding.
        sample_frac:
            If set, randomly sample this fraction of the data (0 < frac ≤ 1).
        random_state:
            Random seed for reproducibility.

        Returns
        -------
        (DataFrame, DatasetValidationResponse)
            The DataFrame has columns from ``FEATURE_SCHEMA`` + ``"label"``.
        """
        file_path = Path(file_path)
        if not file_path.exists():
            raise FileNotFoundError(f"Dataset file not found: {file_path}")

        logger.info("Loading dataset from %s", file_path)
        df = pd.read_csv(file_path, encoding="utf-8", low_memory=False)

        # Strip whitespace from column names (CIC-IDS2017 has leading spaces)
        df.columns = df.columns.str.strip()

        # Auto-detect dataset type
        if dataset_type == "auto":
            dataset_type = self._detect_dataset_type(df)
            logger.info("Auto-detected dataset type: %s", dataset_type)

        # Validate before mapping
        warnings: list[str] = []
        raw_missing = df.isna().sum()
        missing_dict = {c: int(v) for c, v in raw_missing.items() if v > 0}
        duplicate_rows = int(df.duplicated().sum())

        if duplicate_rows > 0:
            warnings.append(f"Found {duplicate_rows} duplicate rows")

        # Map columns
        if dataset_type == "cic-ids2017":
            df, map_warnings = self._map_cic_ids2017(df, task)
        elif dataset_type == "unsw-nb15":
            df, map_warnings = self._map_unsw_nb15(df, task)
        else:
            df, map_warnings = self._map_generic(df)

        warnings.extend(map_warnings)

        # Sample if requested
        if sample_frac is not None and 0 < sample_frac < 1.0:
            df = df.sample(frac=sample_frac, random_state=random_state).reset_index(
                drop=True
            )
            logger.info("Sampled %.1f%% → %d rows", sample_frac * 100, len(df))

        # Class distribution
        class_dist: dict[str, int] = {}
        if "label" in df.columns:
            class_dist = df["label"].value_counts().to_dict()
            class_dist = {str(k): int(v) for k, v in class_dist.items()}

        # Feature mapping status
        mapped_features = [c for c in FEATURE_SCHEMA if c in df.columns]
        missing_features = [c for c in FEATURE_SCHEMA if c not in df.columns]
        if missing_features:
            warnings.append(
                f"{len(missing_features)} schema features not mapped: "
                f"{missing_features[:10]}{'...' if len(missing_features) > 10 else ''}"
            )
        mapping_status = (
            "complete"
            if not missing_features
            else f"partial ({len(mapped_features)}/{len(FEATURE_SCHEMA)})"
        )

        # Build validation response
        validation = DatasetValidationResponse(
            name=file_path.stem,
            rows=len(df),
            columns=len(df.columns),
            column_names=list(df.columns),
            missing_values=missing_dict,
            duplicate_rows=duplicate_rows,
            class_distribution=class_dist,
            feature_mapping_status=mapping_status,
            warnings=warnings,
            preprocessing_summary={
                "dataset_type": dataset_type,
                "task": task,
                "mapped_features": len(mapped_features),
                "total_schema_features": len(FEATURE_SCHEMA),
                "original_columns": len(raw_missing),
            },
        )

        logger.info(
            "Dataset loaded: %d rows, %d cols, mapping=%s, %d warnings",
            len(df),
            len(df.columns),
            mapping_status,
            len(warnings),
        )

        return df, validation

    def load_multiple(
        self,
        file_paths: list[str | Path],
        dataset_type: str = "auto",
        *,
        task: str = "binary",
    ) -> tuple[pd.DataFrame, list["DatasetValidationResponse"]]:
        """Load and concatenate multiple CSVs of the **same** dataset type.

        Parameters
        ----------
        file_paths:
            List of CSV file paths (all must be the same dataset type).
        dataset_type:
            Dataset type (applied to all files).
        task:
            Label encoding task.

        Returns
        -------
        (combined DataFrame, list of validation responses)

        .. warning::

           This concatenates files from the **same** dataset only. It does NOT
           blindly combine CIC-IDS2017 and UNSW-NB15 — their feature spaces
           and label semantics differ.
        """
        dfs: list[pd.DataFrame] = []
        validations: list[DatasetValidationResponse] = []

        for fp in file_paths:
            df, val = self.load(fp, dataset_type=dataset_type, task=task)
            dfs.append(df)
            validations.append(val)

        if not dfs:
            raise ValueError("No files provided to load_multiple()")

        combined = pd.concat(dfs, ignore_index=True)
        logger.info(
            "Combined %d files → %d total rows", len(file_paths), len(combined)
        )
        return combined, validations

    def validate(self, file_path: str | Path) -> "DatasetValidationResponse":
        """Validate a dataset file without fully loading it.

        Returns a ``DatasetValidationResponse`` with column info, missing
        values, class distribution, and warnings.
        """
        _, validation = self.load(file_path, dataset_type="auto", task="binary")
        return validation

    # ------------------------------------------------------------------
    # Dataset-specific mapping
    # ------------------------------------------------------------------

    def _map_cic_ids2017(
        self, df: pd.DataFrame, task: str
    ) -> tuple[pd.DataFrame, list[str]]:
        """Map CIC-IDS2017 columns to FEATURE_SCHEMA."""
        warnings: list[str] = []

        # Build rename map (only for columns that exist)
        rename_map: dict[str, str] = {}
        for orig, schema_name in CIC_IDS2017_COLUMN_MAP.items():
            if orig in df.columns:
                rename_map[orig] = schema_name

        unmapped = [
            c
            for c in df.columns
            if c not in rename_map and c not in _EXCLUDED_COLUMNS_SET
        ]
        if unmapped:
            logger.debug("CIC-IDS2017 unmapped columns: %s", unmapped[:20])

        df = df.rename(columns=rename_map)

        # Encode labels
        if "label" in df.columns:
            df = self._encode_labels_cic(df, task)
        else:
            warnings.append("No 'Label'/' Label' column found in CIC-IDS2017 data")

        # Drop non-feature columns
        df = self._keep_schema_columns(df)

        return df, warnings

    def _map_unsw_nb15(
        self, df: pd.DataFrame, task: str
    ) -> tuple[pd.DataFrame, list[str]]:
        """Map UNSW-NB15 columns to FEATURE_SCHEMA."""
        warnings: list[str] = []

        # Lowercase column names for consistent matching
        df.columns = df.columns.str.lower().str.strip()

        rename_map: dict[str, str] = {}
        for orig, schema_name in UNSW_NB15_COLUMN_MAP.items():
            orig_lower = orig.lower()
            if orig_lower in df.columns:
                rename_map[orig_lower] = schema_name

        df = df.rename(columns=rename_map)

        # Encode labels
        if "label" in df.columns:
            if task == "multiclass" and "attack_category" in df.columns:
                df = self._encode_labels_unsw_multi(df)
            else:
                # Binary: label column is already 0/1 in UNSW-NB15
                df["label"] = df["label"].astype(int)
        else:
            warnings.append("No 'label' column found in UNSW-NB15 data")

        # Encode protocol if it is still a string
        if "protocol_type" in df.columns:
            proto_map = {"tcp": 6, "udp": 17, "icmp": 1}
            df["protocol_type"] = (
                df["protocol_type"]
                .astype(str)
                .str.lower()
                .map(proto_map)
                .fillna(0)
                .astype(float)
            )

        # Drop attack_category helper column
        if "attack_category" in df.columns:
            df = df.drop(columns=["attack_category"])

        df = self._keep_schema_columns(df)

        return df, warnings

    def _map_generic(
        self, df: pd.DataFrame
    ) -> tuple[pd.DataFrame, list[str]]:
        """Passthrough for CSVs already using FEATURE_SCHEMA column names."""
        warnings: list[str] = []
        present = [c for c in FEATURE_SCHEMA if c in df.columns]
        if len(present) < len(FEATURE_SCHEMA) // 2:
            warnings.append(
                f"Generic mode: only {len(present)}/{len(FEATURE_SCHEMA)} "
                f"schema features found. Consider using a dataset-specific loader."
            )
        df = self._keep_schema_columns(df)
        return df, warnings

    # ------------------------------------------------------------------
    # Label encoding
    # ------------------------------------------------------------------

    @staticmethod
    def _encode_labels_cic(df: pd.DataFrame, task: str) -> pd.DataFrame:
        """Encode CIC-IDS2017 labels for binary or multiclass."""
        df["label"] = df["label"].astype(str).str.strip()
        if task == "binary":
            df["label"] = df["label"].apply(
                lambda x: 0 if x == "BENIGN" else 1
            )
        else:
            # Multiclass
            df["label"] = df["label"].apply(
                lambda x: _CIC_MULTI_MAP.get(x, len(_CIC_MULTI_MAP))
            )
        return df

    @staticmethod
    def _encode_labels_unsw_multi(df: pd.DataFrame) -> pd.DataFrame:
        """Encode UNSW-NB15 attack_category for multiclass."""
        df["attack_category"] = df["attack_category"].astype(str).str.strip()
        df["label"] = df["attack_category"].apply(
            lambda x: _UNSW_MULTI_MAP.get(x, len(_UNSW_MULTI_MAP))
        )
        return df

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _detect_dataset_type(df: pd.DataFrame) -> str:
        """Heuristically detect the dataset type from column names."""
        cols_set = set(df.columns)

        # CIC-IDS2017 signatures
        cic_markers = {"Flow Duration", "Total Fwd Packets", "Flow Bytes/s", "Label"}
        if cic_markers & cols_set:
            return "cic-ids2017"

        # UNSW-NB15 signatures
        unsw_markers = {"attack_cat", "dur", "sbytes", "dbytes", "spkts"}
        # Also check lowercased
        cols_lower = {c.lower() for c in cols_set}
        unsw_markers_lower = {m.lower() for m in unsw_markers}
        if unsw_markers_lower & cols_lower:
            return "unsw-nb15"

        return "generic"

    @staticmethod
    def _keep_schema_columns(df: pd.DataFrame) -> pd.DataFrame:
        """Keep only FEATURE_SCHEMA columns + 'label', drop everything else."""
        keep = [c for c in df.columns if c in _SCHEMA_SET or c == "label"]
        return df[keep]


# Module-level constants for fast lookups
_SCHEMA_SET = set(FEATURE_SCHEMA)
_EXCLUDED_COLUMNS_SET = frozenset({
    "src_ip", "dst_ip", "timestamp", "flow_id",
    "Flow ID", "Source IP", "Destination IP", "Timestamp",
    "srcip", "dstip",
})
