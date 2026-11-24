"""
SQLAlchemy database models for the NIDS/NIPS system.
Defines all database tables: network flows, incidents, models, anomalies,
Suricata alerts, mitigation actions, settings, and audit logs.
"""
import datetime
import uuid
from sqlalchemy import (
    Column, String, Integer, Float, DateTime, Boolean, Text, JSON,
    ForeignKey, Index, create_engine, Enum as SAEnum
)
from sqlalchemy.orm import declarative_base, relationship, sessionmaker
from backend.core.config import DATABASE_URL

Base = declarative_base()


def generate_uuid() -> str:
    return str(uuid.uuid4())


def utcnow() -> datetime.datetime:
    return datetime.datetime.now(datetime.timezone.utc)


# ───────────────────────── Network Flows ─────────────────────────
class NetworkFlow(Base):
    __tablename__ = "network_flows"

    id = Column(String, primary_key=True, default=generate_uuid)
    flow_id = Column(String, index=True, nullable=True)  # original flow ID from dataset
    timestamp = Column(DateTime, default=utcnow, index=True)
    src_ip = Column(String, nullable=True)
    dst_ip = Column(String, nullable=True)
    src_port = Column(Integer, nullable=True)
    dst_port = Column(Integer, nullable=True)
    protocol = Column(String, nullable=True)
    duration = Column(Float, nullable=True)
    total_fwd_packets = Column(Integer, nullable=True)
    total_bwd_packets = Column(Integer, nullable=True)
    total_bytes = Column(Integer, nullable=True)
    label = Column(String, nullable=True)        # ground truth label
    attack_type = Column(String, nullable=True)   # specific attack category
    source = Column(String, nullable=True)        # e.g. 'cic-ids2017', 'unsw-nb15', 'live', 'pcap'
    features = Column(JSON, nullable=True)        # full extracted feature vector
    created_at = Column(DateTime, default=utcnow)

    incidents = relationship("DetectionIncident", back_populates="flow")

    __table_args__ = (
        Index("ix_flow_src_dst", "src_ip", "dst_ip"),
        Index("ix_flow_label", "label"),
    )


# ───────────────────────── Detection Incidents ─────────────────────────
class DetectionIncident(Base):
    __tablename__ = "detection_incidents"

    id = Column(String, primary_key=True, default=generate_uuid)
    flow_id = Column(String, ForeignKey("network_flows.id"), nullable=True, index=True)
    timestamp = Column(DateTime, default=utcnow, index=True)
    detection_method = Column(String, nullable=False)  # 'signature', 'ml', 'anomaly', 'hybrid'
    predicted_label = Column(String, nullable=True)
    attack_type = Column(String, nullable=True)
    model_confidence = Column(Float, nullable=True)
    anomaly_score = Column(Float, nullable=True)
    risk_score = Column(Float, nullable=True)
    severity = Column(String, nullable=True)       # 'low', 'medium', 'high', 'critical'
    signature_id = Column(String, nullable=True)
    signature_msg = Column(String, nullable=True)
    src_ip = Column(String, nullable=True)
    dst_ip = Column(String, nullable=True)
    src_port = Column(Integer, nullable=True)
    dst_port = Column(Integer, nullable=True)
    protocol = Column(String, nullable=True)
    evidence = Column(JSON, nullable=True)         # supporting evidence dict
    shap_values = Column(JSON, nullable=True)      # SHAP feature contributions
    status = Column(String, default="open")        # open, investigating, resolved, false_positive
    resolved_at = Column(DateTime, nullable=True)
    notes = Column(Text, nullable=True)
    created_at = Column(DateTime, default=utcnow)

    flow = relationship("NetworkFlow", back_populates="incidents")

    __table_args__ = (
        Index("ix_incident_severity", "severity"),
        Index("ix_incident_status", "status"),
        Index("ix_incident_method", "detection_method"),
    )


# ───────────────────────── Model Registry ─────────────────────────
class ModelInfo(Base):
    __tablename__ = "model_info"

    id = Column(String, primary_key=True, default=generate_uuid)
    name = Column(String, unique=True, nullable=False)
    model_type = Column(String, nullable=False)      # 'xgboost', 'random_forest', etc.
    task = Column(String, nullable=False)              # 'binary', 'multiclass'
    dataset = Column(String, nullable=True)
    version = Column(Integer, default=1)
    file_path = Column(String, nullable=True)
    scaler_path = Column(String, nullable=True)
    feature_columns = Column(JSON, nullable=True)
    hyperparameters = Column(JSON, nullable=True)
    training_samples = Column(Integer, nullable=True)
    is_active = Column(Boolean, default=False)
    created_at = Column(DateTime, default=utcnow)

    evaluations = relationship("ModelEvaluation", back_populates="model")


# ───────────────────────── Model Evaluation ─────────────────────────
class ModelEvaluation(Base):
    __tablename__ = "model_evaluations"

    id = Column(String, primary_key=True, default=generate_uuid)
    model_id = Column(String, ForeignKey("model_info.id"), nullable=False, index=True)
    dataset = Column(String, nullable=True)
    task = Column(String, nullable=True)
    accuracy = Column(Float, nullable=True)
    precision_macro = Column(Float, nullable=True)
    recall_macro = Column(Float, nullable=True)
    f1_macro = Column(Float, nullable=True)
    f1_weighted = Column(Float, nullable=True)
    false_positive_rate = Column(Float, nullable=True)
    inference_latency_ms = Column(Float, nullable=True)
    confusion_matrix = Column(JSON, nullable=True)
    classification_report = Column(JSON, nullable=True)
    class_distribution = Column(JSON, nullable=True)
    created_at = Column(DateTime, default=utcnow)

    model = relationship("ModelInfo", back_populates="evaluations")


# ───────────────────────── Anomaly Events ─────────────────────────
class AnomalyEvent(Base):
    __tablename__ = "anomaly_events"

    id = Column(String, primary_key=True, default=generate_uuid)
    flow_id = Column(String, ForeignKey("network_flows.id"), nullable=True)
    timestamp = Column(DateTime, default=utcnow, index=True)
    anomaly_score = Column(Float, nullable=False)
    threshold = Column(Float, nullable=True)
    is_anomaly = Column(Boolean, default=False)
    src_ip = Column(String, nullable=True)
    dst_ip = Column(String, nullable=True)
    protocol = Column(String, nullable=True)
    features_summary = Column(JSON, nullable=True)
    created_at = Column(DateTime, default=utcnow)


# ───────────────────────── Suricata Alerts ─────────────────────────
class SuricataAlert(Base):
    __tablename__ = "suricata_alerts"

    id = Column(String, primary_key=True, default=generate_uuid)
    timestamp = Column(DateTime, default=utcnow, index=True)
    signature_id = Column(Integer, nullable=True)
    signature = Column(String, nullable=True)
    category = Column(String, nullable=True)
    severity = Column(Integer, nullable=True)
    src_ip = Column(String, nullable=True)
    dst_ip = Column(String, nullable=True)
    src_port = Column(Integer, nullable=True)
    dst_port = Column(Integer, nullable=True)
    protocol = Column(String, nullable=True)
    action = Column(String, nullable=True)
    payload = Column(Text, nullable=True)
    flow_id = Column(String, nullable=True)
    raw_event = Column(JSON, nullable=True)
    created_at = Column(DateTime, default=utcnow)


# ───────────────────────── Mitigation Actions ─────────────────────────
class MitigationAction(Base):
    __tablename__ = "mitigation_actions"

    id = Column(String, primary_key=True, default=generate_uuid)
    action_type = Column(String, nullable=False)  # 'block', 'unblock'
    target_ip = Column(String, nullable=False, index=True)
    reason = Column(Text, nullable=True)
    incident_id = Column(String, ForeignKey("detection_incidents.id"), nullable=True)
    duration_minutes = Column(Integer, nullable=True)
    expires_at = Column(DateTime, nullable=True)
    is_active = Column(Boolean, default=True)
    is_simulated = Column(Boolean, default=True)
    approved_by = Column(String, nullable=True)
    nftables_handle = Column(String, nullable=True)
    status = Column(String, default="active")  # active, expired, removed, failed
    error_message = Column(Text, nullable=True)
    created_at = Column(DateTime, default=utcnow)
    removed_at = Column(DateTime, nullable=True)


# ───────────────────────── Admin Settings ─────────────────────────
class AdminSetting(Base):
    __tablename__ = "admin_settings"

    id = Column(String, primary_key=True, default=generate_uuid)
    key = Column(String, unique=True, nullable=False, index=True)
    value = Column(JSON, nullable=True)
    description = Column(Text, nullable=True)
    updated_at = Column(DateTime, default=utcnow)


# ───────────────────────── Audit Logs ─────────────────────────
class AuditLog(Base):
    __tablename__ = "audit_logs"

    id = Column(String, primary_key=True, default=generate_uuid)
    timestamp = Column(DateTime, default=utcnow, index=True)
    action = Column(String, nullable=False)
    module = Column(String, nullable=True)
    details = Column(JSON, nullable=True)
    user = Column(String, nullable=True)
    ip_address = Column(String, nullable=True)
    success = Column(Boolean, default=True)
    created_at = Column(DateTime, default=utcnow)


# ───────────────────────── Database Setup ─────────────────────────
engine = create_engine(DATABASE_URL, echo=False, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(bind=engine, autocommit=False, autoflush=False)


def init_db():
    """Create all tables."""
    Base.metadata.create_all(bind=engine)


def get_db():
    """Get a database session. Use as a context manager or dependency."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
