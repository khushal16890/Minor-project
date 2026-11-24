"""
Dashboard overview API router.
"""
from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session
from sqlalchemy import func
from backend.database.models import (
    get_db, NetworkFlow, DetectionIncident, MitigationAction
)
from backend.schemas.schemas import DashboardOverview

router = APIRouter()


@router.get("/dashboard/overview", response_model=DashboardOverview)
def get_dashboard_overview(db: Session = Depends(get_db)):
    total_flows = db.query(NetworkFlow).count()
    benign_flows = db.query(NetworkFlow).filter(
        (NetworkFlow.label == "BENIGN") | (NetworkFlow.label == "0") | (NetworkFlow.label == "benign")
    ).count()
    malicious_flows = total_flows - benign_flows

    active_incidents = db.query(DetectionIncident).filter(
        DetectionIncident.status.in_(["open", "investigating"])
    ).count()

    critical_incidents = db.query(DetectionIncident).filter(
        DetectionIncident.severity == "critical",
        DetectionIncident.status.in_(["open", "investigating"])
    ).count()

    active_blocks = db.query(MitigationAction).filter(
        MitigationAction.is_active == True,
        MitigationAction.action_type == "block"
    ).count()

    # Attack distribution
    attack_counts = (
        db.query(DetectionIncident.attack_type, func.count(DetectionIncident.id))
        .filter(DetectionIncident.attack_type != None)
        .group_by(DetectionIncident.attack_type)
        .all()
    )
    attack_distribution = {atk: cnt for atk, cnt in attack_counts if atk}

    # Detection method counts
    method_counts = (
        db.query(DetectionIncident.detection_method, func.count(DetectionIncident.id))
        .group_by(DetectionIncident.detection_method)
        .all()
    )
    detection_method_counts = {method: cnt for method, cnt in method_counts if method}

    # Recent events
    recent_incidents = (
        db.query(DetectionIncident)
        .order_by(DetectionIncident.created_at.desc())
        .limit(10)
        .all()
    )
    recent_events = [
        {
            "id": inc.id,
            "timestamp": inc.timestamp.isoformat() if inc.timestamp else None,
            "attack_type": inc.attack_type or "Unknown",
            "severity": inc.severity or "low",
            "detection_method": inc.detection_method,
            "src_ip": inc.src_ip or "N/A",
            "dst_ip": inc.dst_ip or "N/A",
            "risk_score": inc.risk_score or 0.0,
            "status": inc.status,
        }
        for inc in recent_incidents
    ]

    return DashboardOverview(
        total_flows=total_flows,
        benign_flows=benign_flows,
        malicious_flows=malicious_flows,
        active_incidents=active_incidents,
        critical_incidents=critical_incidents,
        active_blocks=active_blocks,
        recent_events=recent_events,
        attack_distribution=attack_distribution,
        detection_method_counts=detection_method_counts,
    )
