"""
System settings API router.
"""
from fastapi import APIRouter
from backend.core.config import get_config, update_config, AppConfig
from backend.schemas.schemas import SettingsResponse, SettingsUpdateRequest

router = APIRouter()


@router.get("/settings", response_model=SettingsResponse)
def get_settings():
    cfg = get_config()
    return SettingsResponse(
        risk_weights=cfg.risk_weights.model_dump(),
        risk_thresholds=cfg.risk_thresholds.model_dump(),
        mitigation=cfg.mitigation.model_dump(),
        detection=cfg.detection.model_dump(),
        network=cfg.network.model_dump(),
        active_model=cfg.active_model
    )


@router.post("/settings", response_model=SettingsResponse)
def update_settings(req: SettingsUpdateRequest):
    cfg = get_config()
    if req.risk_weights:
        cfg.risk_weights = cfg.risk_weights.model_copy(update=req.risk_weights)
    if req.risk_thresholds:
        cfg.risk_thresholds = cfg.risk_thresholds.model_copy(update=req.risk_thresholds)
    if req.mitigation:
        cfg.mitigation = cfg.mitigation.model_copy(update=req.mitigation)
    if req.detection:
        cfg.detection = cfg.detection.model_copy(update=req.detection)
    if req.network:
        cfg.network = cfg.network.model_copy(update=req.network)
    if req.active_model is not None:
        cfg.active_model = req.active_model

    update_config(cfg)
    return get_settings()
