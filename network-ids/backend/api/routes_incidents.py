"""
Incident management API router.
"""
from typing import Optional
import datetime
from fastapi import APIRouter, Depends, Query, HTTPException
from sqlalchemy.orm import Session
from backend.database.models import get_db, DetectionIncident, utcnow
from backend.schemas.schemas import IncidentListResponse, IncidentResponse

router = APIRouter()


@router.get("/incidents", response_model=IncidentListResponse)
def list_incidents(
    page: int = Query(1, ge=1),
    page_size: int = Query(50, ge=1, le=500),
    severity: Optional[str] = None,
    detection_method: Optional[str] = None,
    status: Optional[str] = None,
    db: Session = Depends(get_db)
):
    query = db.query(DetectionIncident)
    if severity:
        query = query.filter(DetectionIncident.severity == severity.lower())
    if detection_method:
        query = query.filter(DetectionIncident.detection_method == detection_method.lower())
    if status:
        query = query.filter(DetectionIncident.status == status.lower())

    total = query.count()
    incidents = (
        query.order_by(DetectionIncident.created_at.desc())
        .offset((page - 1) * page_size)
        .limit(page_size)
        .all()
    )
    return IncidentListResponse(
        total=total,
        page=page,
        page_size=page_size,
        incidents=[IncidentResponse.model_validate(inc) for inc in incidents]
    )


@router.get("/incidents/{incident_id}", response_model=IncidentResponse)
def get_incident(incident_id: str, db: Session = Depends(get_db)):
    incident = db.query(DetectionIncident).filter(DetectionIncident.id == incident_id).first()
    if not incident:
        raise HTTPException(status_code=404, detail="Incident not found")
    return IncidentResponse.model_validate(incident)


@router.post("/incidents/{incident_id}/resolve")
def resolve_incident(incident_id: str, status: str = "resolved", notes: Optional[str] = None, db: Session = Depends(get_db)):
    incident = db.query(DetectionIncident).filter(DetectionIncident.id == incident_id).first()
    if not incident:
        raise HTTPException(status_code=404, detail="Incident not found")
    incident.status = status
    if notes:
        incident.notes = notes
    incident.resolved_at = utcnow()
    db.commit()
    return {"message": f"Incident {incident_id} updated to {status}"}
