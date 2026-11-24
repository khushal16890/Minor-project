"""
Pytest configuration and shared fixtures for the test suite.
"""
import pytest
import pandas as pd
import numpy as np
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from backend.database.models import Base
from backend.core.config import FEATURE_SCHEMA


@pytest.fixture
def sample_feature_dict():
    """Returns a dict containing a sample network flow feature vector."""
    return {
        "flow_duration": 1.25,
        "protocol_type": "tcp",
        "src_port": 44321,
        "dst_port": 80,
        "fwd_packets": 12,
        "bwd_packets": 8,
        "total_fwd_bytes": 1450,
        "total_bwd_bytes": 4820,
        "fwd_pkt_len_max": 1400.0,
        "fwd_pkt_len_min": 40.0,
        "fwd_pkt_len_mean": 120.8,
        "fwd_pkt_len_std": 210.5,
        "bwd_pkt_len_max": 1460.0,
        "bwd_pkt_len_min": 40.0,
        "bwd_pkt_len_mean": 602.5,
        "bwd_pkt_len_std": 450.2,
        "flow_bytes_per_sec": 5016.0,
        "flow_pkts_per_sec": 16.0,
        "flow_iat_mean": 0.062,
        "flow_iat_std": 0.045,
        "flow_iat_max": 0.25,
        "flow_iat_min": 0.001,
        "fwd_iat_mean": 0.10,
        "fwd_iat_std": 0.05,
        "fwd_iat_max": 0.30,
        "fwd_iat_min": 0.002,
        "bwd_iat_mean": 0.15,
        "bwd_iat_std": 0.06,
        "bwd_iat_max": 0.40,
        "bwd_iat_min": 0.005,
        "fwd_psh_flags": 1,
        "bwd_psh_flags": 0,
        "fwd_urg_flags": 0,
        "bwd_urg_flags": 0,
        "fin_flag_cnt": 0,
        "syn_flag_cnt": 1,
        "rst_flag_cnt": 0,
        "psh_flag_cnt": 1,
        "ack_flag_cnt": 1,
        "urg_flag_cnt": 0,
        "ece_flag_cnt": 0,
        "cwe_flag_cnt": 0,
        "pkt_len_mean": 313.5,
        "pkt_len_std": 380.0,
        "pkt_len_var": 144400.0,
        "avg_pkt_size": 313.5,
        "fwd_seg_size_avg": 120.8,
        "bwd_seg_size_avg": 602.5,
        "subflow_fwd_pkts": 12,
        "subflow_bwd_pkts": 8,
        "subflow_fwd_bytes": 1450,
        "subflow_bwd_bytes": 4820,
        "init_win_bytes_fwd": 29200,
        "init_win_bytes_bwd": 28960,
        "act_data_pkt_fwd": 8,
        "min_seg_size_fwd": 20,
        "active_mean": 1.0,
        "active_std": 0.0,
        "active_max": 1.0,
        "active_min": 1.0,
        "idle_mean": 0.25,
        "idle_std": 0.0,
        "idle_max": 0.25,
        "idle_min": 0.25,
        "conn_frequency": 3,
    }


@pytest.fixture
def sample_df(sample_feature_dict):
    """Returns a synthetic DataFrame with benign and malicious samples."""
    rows = []
    for i in range(100):
        feat = sample_feature_dict.copy()
        # Vary numerical values
        feat["flow_duration"] += np.random.uniform(-0.5, 0.5)
        feat["total_fwd_bytes"] += int(np.random.uniform(-200, 200))
        feat["Label"] = 0 if i < 70 else 1
        rows.append(feat)
    return pd.DataFrame(rows)


@pytest.fixture
def in_memory_db():
    """Provides an isolated in-memory SQLite database session."""
    engine = create_engine("sqlite:///:memory:", echo=False)
    Base.metadata.create_all(bind=engine)
    Session = sessionmaker(bind=engine)
    session = Session()
    try:
        yield session
    finally:
        session.close()
