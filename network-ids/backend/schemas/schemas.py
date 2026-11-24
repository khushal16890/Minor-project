"""
Pydantic schemas for API request/response validation.
Separates API contracts from database models.
"""
import datetime
from typing import Optional, Any
from pydantic import BaseModel, Field


# ───────────────────────── Health ─────────────────────────
class HealthResponse(BaseModel):
    status: str = "ok"
    version: str = "1.0.0"
    database: str = "connected"
    models_loaded: int = 0
    uptime_seconds: float = 0.0


# ───────────────────────── Dashboard ─────────────────────────
class DashboardOverview(BaseModel):
    total_flows: int = 0
    benign_flows: int = 0
    malicious_flows: int = 0
    active_incidents: int = 0
    critical_incidents: int = 0
    active_blocks: int = 0
    recent_events: list[dict] = Field(default_factory=list)
    attack_distribution: dict[str, int] = Field(default_factory=dict)
    detection_method_counts: dict[str, int] = Field(default_factory=dict)


# ───────────────────────── Traffic / Flows ─────────────────────────
class FlowResponse(BaseModel):
    id: str
    flow_id: Optional[str] = None
    timestamp: Optional[datetime.datetime] = None
    src_ip: Optional[str] = None
    dst_ip: Optional[str] = None
    src_port: Optional[int] = None
    dst_port: Optional[int] = None
    protocol: Optional[str] = None
    duration: Optional[float] = None
    total_bytes: Optional[int] = None
    label: Optional[str] = None
    attack_type: Optional[str] = None
    source: Optional[str] = None

    class Config:
        from_attributes = True


class FlowListResponse(BaseModel):
    total: int
    page: int
    page_size: int
    flows: list[FlowResponse]


class TrafficAnalysisRequest(BaseModel):
    features: dict[str, float]


class TrafficAnalysisResponse(BaseModel):
    predicted_label: str
    attack_type: Optional[str] = None
    confidence: float
    anomaly_score: Optional[float] = None
    risk_score: Optional[float] = None
    severity: Optional[str] = None
    shap_values: Optional[dict[str, float]] = None


# ───────────────────────── Incidents ─────────────────────────
class IncidentResponse(BaseModel):
    id: str
    flow_id: Optional[str] = None
    timestamp: Optional[datetime.datetime] = None
    detection_method: str
    predicted_label: Optional[str] = None
    attack_type: Optional[str] = None
    model_confidence: Optional[float] = None
    anomaly_score: Optional[float] = None
    risk_score: Optional[float] = None
    severity: Optional[str] = None
    signature_id: Optional[str] = None
    signature_msg: Optional[str] = None
    src_ip: Optional[str] = None
    dst_ip: Optional[str] = None
    src_port: Optional[int] = None
    dst_port: Optional[int] = None
    protocol: Optional[str] = None
    evidence: Optional[dict] = None
    shap_values: Optional[dict[str, float]] = None
    status: str = "open"
    notes: Optional[str] = None
    created_at: Optional[datetime.datetime] = None

    class Config:
        from_attributes = True


class IncidentListResponse(BaseModel):
    total: int
    page: int
    page_size: int
    incidents: list[IncidentResponse]


# ───────────────────────── Models ─────────────────────────
class ModelInfoResponse(BaseModel):
    id: str
    name: str
    model_type: str
    task: str
    dataset: Optional[str] = None
    version: int = 1
    is_active: bool = False
    training_samples: Optional[int] = None
    created_at: Optional[datetime.datetime] = None

    class Config:
        from_attributes = True


class ModelEvalResponse(BaseModel):
    id: str
    model_id: str
    dataset: Optional[str] = None
    task: Optional[str] = None
    accuracy: Optional[float] = None
    precision_macro: Optional[float] = None
    recall_macro: Optional[float] = None
    f1_macro: Optional[float] = None
    f1_weighted: Optional[float] = None
    false_positive_rate: Optional[float] = None
    inference_latency_ms: Optional[float] = None
    confusion_matrix: Optional[list] = None
    classification_report: Optional[dict] = None
    created_at: Optional[datetime.datetime] = None

    class Config:
        from_attributes = True


class TrainRequest(BaseModel):
    model_type: str = Field(description="xgboost, random_forest, logistic_regression, lightgbm")
    task: str = Field(default="binary", description="binary or multiclass")
    dataset_name: str = Field(description="Name of the preprocessed dataset to train on")
    test_size: float = Field(default=0.2, ge=0.1, le=0.5)
    hyperparameters: Optional[dict] = None


class EvaluateRequest(BaseModel):
    model_name: str
    dataset_name: str
    task: Optional[str] = None


# ───────────────────────── Dataset ─────────────────────────
class DatasetValidationResponse(BaseModel):
    name: str
    rows: int
    columns: int
    column_names: list[str]
    missing_values: dict[str, int]
    duplicate_rows: int
    class_distribution: dict[str, int]
    feature_mapping_status: str
    warnings: list[str] = Field(default_factory=list)
    preprocessing_summary: dict[str, Any] = Field(default_factory=dict)


class DatasetUploadResponse(BaseModel):
    name: str
    file_path: str
    rows: int
    columns: int
    message: str


# ───────────────────────── PCAP ─────────────────────────
class PcapAnalysisResponse(BaseModel):
    file_name: str
    total_packets: int
    flows_extracted: int
    incidents_detected: int
    summary: dict[str, Any] = Field(default_factory=dict)


# ───────────────────────── Mitigation ─────────────────────────
class BlockRequest(BaseModel):
    target_ip: str
    reason: str
    duration_minutes: int = Field(default=60, ge=1, le=1440)
    incident_id: Optional[str] = None


class UnblockRequest(BaseModel):
    target_ip: str
    reason: Optional[str] = None


class MitigationActionResponse(BaseModel):
    id: str
    action_type: str
    target_ip: str
    reason: Optional[str] = None
    duration_minutes: Optional[int] = None
    expires_at: Optional[datetime.datetime] = None
    is_active: bool
    is_simulated: bool
    status: str
    created_at: Optional[datetime.datetime] = None

    class Config:
        from_attributes = True


class MitigationListResponse(BaseModel):
    total: int
    actions: list[MitigationActionResponse]


# ───────────────────────── Settings ─────────────────────────
class SettingsResponse(BaseModel):
    risk_weights: dict[str, float]
    risk_thresholds: dict[str, int]
    mitigation: dict[str, Any]
    detection: dict[str, Any]
    network: dict[str, Any]
    active_model: Optional[str] = None


class SettingsUpdateRequest(BaseModel):
    risk_weights: Optional[dict[str, float]] = None
    risk_thresholds: Optional[dict[str, int]] = None
    mitigation: Optional[dict[str, Any]] = None
    detection: Optional[dict[str, Any]] = None
    network: Optional[dict[str, Any]] = None
    active_model: Optional[str] = None
