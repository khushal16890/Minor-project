"""
Application configuration module.
Central configuration for all system components.
"""
import os
from pathlib import Path
from pydantic import BaseModel, Field
from typing import Optional


# Base paths
PROJECT_ROOT = Path(__file__).parent.parent.parent
DATA_DIR = PROJECT_ROOT / "datasets"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
MODELS_DIR = PROJECT_ROOT / "models"
EXPERIMENTS_DIR = PROJECT_ROOT / "experiments"
DB_DIR = PROJECT_ROOT / "backend" / "database"

# Ensure directories exist
for d in [RAW_DATA_DIR, PROCESSED_DATA_DIR, MODELS_DIR, EXPERIMENTS_DIR, DB_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# Database
DATABASE_URL = os.getenv("DATABASE_URL", f"sqlite:///{DB_DIR / 'nids.db'}")

# API
API_HOST = os.getenv("API_HOST", "127.0.0.1")
API_PORT = int(os.getenv("API_PORT", "8000"))
API_SECRET_KEY = os.getenv("API_SECRET_KEY", "change-me-in-production")

# Dashboard
DASHBOARD_HOST = os.getenv("DASHBOARD_HOST", "127.0.0.1")
DASHBOARD_PORT = int(os.getenv("DASHBOARD_PORT", "8501"))


class RiskWeights(BaseModel):
    """Configurable risk scoring weights. Must sum to 1.0."""
    model_confidence: float = Field(default=0.40, ge=0, le=1)
    attack_severity: float = Field(default=0.30, ge=0, le=1)
    anomaly_score: float = Field(default=0.20, ge=0, le=1)
    frequency_score: float = Field(default=0.10, ge=0, le=1)


class RiskThresholds(BaseModel):
    """Risk level boundaries."""
    low_max: int = 29
    medium_max: int = 59
    high_max: int = 79
    # 80-100 is critical


class MitigationConfig(BaseModel):
    """Mitigation engine configuration."""
    enabled: bool = Field(default=False, description="Master switch for mitigation")
    auto_block_critical: bool = Field(default=False, description="Auto-block on critical threats")
    simulation_mode: bool = Field(default=True, description="Simulate without actual firewall changes")
    default_block_duration_minutes: int = Field(default=60, ge=1, le=1440)
    protected_ips: list[str] = Field(default_factory=lambda: ["127.0.0.1", "::1"])
    management_interface: Optional[str] = Field(default=None, description="Network interface to never block")
    max_active_blocks: int = Field(default=100, ge=1)


class DetectionConfig(BaseModel):
    """Detection engine configuration."""
    anomaly_threshold: float = Field(default=-0.5, description="Isolation Forest anomaly threshold")
    ml_confidence_threshold: float = Field(default=0.7, ge=0, le=1)
    signature_weight: float = Field(default=0.90, ge=0, le=1, description="Weight for signature detections")
    ml_weight: float = Field(default=0.70, ge=0, le=1, description="Weight for ML detections")
    anomaly_weight: float = Field(default=0.50, ge=0, le=1, description="Weight for anomaly detections")


class NetworkConfig(BaseModel):
    """Network monitoring configuration."""
    capture_interface: Optional[str] = Field(default=None)
    capture_filter: str = Field(default="tcp or udp or icmp")
    pcap_max_size_mb: int = Field(default=100, ge=1)
    zeek_log_dir: Optional[str] = Field(default=None)
    suricata_eve_path: Optional[str] = Field(default=None)


class AppConfig(BaseModel):
    """Master application configuration."""
    risk_weights: RiskWeights = Field(default_factory=RiskWeights)
    risk_thresholds: RiskThresholds = Field(default_factory=RiskThresholds)
    mitigation: MitigationConfig = Field(default_factory=MitigationConfig)
    detection: DetectionConfig = Field(default_factory=DetectionConfig)
    network: NetworkConfig = Field(default_factory=NetworkConfig)
    active_model: Optional[str] = Field(default=None, description="Currently loaded model name")


# Global configuration instance
_config = AppConfig()


def get_config() -> AppConfig:
    """Get the current application configuration."""
    return _config


def update_config(new_config: AppConfig) -> AppConfig:
    """Update the application configuration."""
    global _config
    _config = new_config
    return _config


# Attack severity mapping (0-100 scale)
ATTACK_SEVERITY = {
    "benign": 0,
    "normal": 0,
    "dos": 70,
    "ddos": 85,
    "portscan": 40,
    "port_scan": 40,
    "brute_force": 60,
    "bruteforce": 60,
    "web_attack": 75,
    "web attack": 75,
    "botnet": 90,
    "infiltration": 80,
    "heartbleed": 95,
    "ssh-patator": 60,
    "ftp-patator": 60,
    "dos slowloris": 65,
    "dos slowhttptest": 65,
    "dos hulk": 70,
    "dos goldeneye": 70,
    "generic": 50,
    "exploits": 80,
    "fuzzers": 45,
    "reconnaissance": 35,
    "analysis": 30,
    "backdoor": 90,
    "shellcode": 85,
    "worms": 90,
    "unknown": 50,
}


def get_attack_severity(attack_type: str) -> int:
    """Get the severity score for an attack type."""
    if not attack_type:
        return 0
    key = attack_type.lower().strip()
    return ATTACK_SEVERITY.get(key, 50)


# Feature schema — the canonical set of features used by ML models
FEATURE_SCHEMA = [
    "flow_duration",
    "protocol_type",
    "src_port",
    "dst_port",
    "fwd_packets",
    "bwd_packets",
    "total_fwd_bytes",
    "total_bwd_bytes",
    "fwd_pkt_len_max",
    "fwd_pkt_len_min",
    "fwd_pkt_len_mean",
    "fwd_pkt_len_std",
    "bwd_pkt_len_max",
    "bwd_pkt_len_min",
    "bwd_pkt_len_mean",
    "bwd_pkt_len_std",
    "flow_bytes_per_sec",
    "flow_pkts_per_sec",
    "flow_iat_mean",
    "flow_iat_std",
    "flow_iat_max",
    "flow_iat_min",
    "fwd_iat_mean",
    "fwd_iat_std",
    "fwd_iat_max",
    "fwd_iat_min",
    "bwd_iat_mean",
    "bwd_iat_std",
    "bwd_iat_max",
    "bwd_iat_min",
    "fwd_psh_flags",
    "bwd_psh_flags",
    "fwd_urg_flags",
    "bwd_urg_flags",
    "fin_flag_cnt",
    "syn_flag_cnt",
    "rst_flag_cnt",
    "psh_flag_cnt",
    "ack_flag_cnt",
    "urg_flag_cnt",
    "ece_flag_cnt",
    "cwe_flag_cnt",
    "pkt_len_mean",
    "pkt_len_std",
    "pkt_len_var",
    "avg_pkt_size",
    "fwd_seg_size_avg",
    "bwd_seg_size_avg",
    "subflow_fwd_pkts",
    "subflow_bwd_pkts",
    "subflow_fwd_bytes",
    "subflow_bwd_bytes",
    "init_win_bytes_fwd",
    "init_win_bytes_bwd",
    "act_data_pkt_fwd",
    "min_seg_size_fwd",
    "active_mean",
    "active_std",
    "active_max",
    "active_min",
    "idle_mean",
    "idle_std",
    "idle_max",
    "idle_min",
    "conn_frequency",
]
