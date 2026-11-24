"""
Mitigation and firewall management API router.
"""
from typing import List
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from backend.database.models import get_db, MitigationAction, AuditLog, utcnow
from backend.schemas.schemas import (
    BlockRequest, UnblockRequest, MitigationActionResponse, MitigationListResponse
)
from mitigation.nftables import NftablesManager
from mitigation.audit import AuditLogger

router = APIRouter()
nft_manager = NftablesManager()
audit_logger = AuditLogger()


@router.get("/mitigation/active", response_model=MitigationListResponse)
def get_active_blocks(db: Session = Depends(get_db)):
    # Sync with nft_manager
    nft_manager.expire_blocks()
    blocks = (
        db.query(MitigationAction)
        .filter(MitigationAction.is_active == True, MitigationAction.action_type == "block")
        .order_by(MitigationAction.created_at.desc())
        .all()
    )
    return MitigationListResponse(
        total=len(blocks),
        actions=[MitigationActionResponse.model_validate(b) for b in blocks]
    )


@router.get("/mitigation/history", response_model=MitigationListResponse)
def get_mitigation_history(db: Session = Depends(get_db)):
    actions = (
        db.query(MitigationAction)
        .order_by(MitigationAction.created_at.desc())
        .limit(100)
        .all()
    )
    return MitigationListResponse(
        total=len(actions),
        actions=[MitigationActionResponse.model_validate(a) for a in actions]
    )


@router.post("/mitigation/block", response_model=MitigationActionResponse)
def block_ip(req: BlockRequest, db: Session = Depends(get_db)):
    res = nft_manager.block_ip(
        ip=req.target_ip,
        reason=req.reason,
        duration_minutes=req.duration_minutes,
        incident_id=req.incident_id,
    )
    if not res.get("success", False):
        raise HTTPException(status_code=400, detail=res.get("message", "Failed to block IP"))

    action_record = (
        db.query(MitigationAction)
        .filter(MitigationAction.target_ip == req.target_ip, MitigationAction.action_type == "block")
        .order_by(MitigationAction.created_at.desc())
        .first()
    )
    if not action_record:
        raise HTTPException(status_code=500, detail="Mitigation action recorded could not be retrieved")

    return MitigationActionResponse.model_validate(action_record)


@router.post("/mitigation/unblock", response_model=MitigationActionResponse)
def unblock_ip(req: UnblockRequest, db: Session = Depends(get_db)):
    res = nft_manager.unblock_ip(
        ip=req.target_ip,
        reason=req.reason or "Manual unblock by administrator",
    )
    if not res.get("success", False):
        raise HTTPException(status_code=400, detail=res.get("message", "Failed to unblock IP"))

    action_record = (
        db.query(MitigationAction)
        .filter(MitigationAction.target_ip == req.target_ip, MitigationAction.action_type == "unblock")
        .order_by(MitigationAction.created_at.desc())
        .first()
    )
    if not action_record:
        action_record = (
            db.query(MitigationAction)
            .filter(MitigationAction.target_ip == req.target_ip)
            .order_by(MitigationAction.created_at.desc())
            .first()
        )

    return MitigationActionResponse.model_validate(action_record)
