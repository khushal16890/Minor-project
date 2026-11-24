"""
Traffic management and flow analysis API router.
"""
from typing import Optional
from fastapi import APIRouter, Depends, Query, HTTPException
from sqlalchemy.orm import Session
from backend.database.models import get_db, NetworkFlow
from backend.schemas.schemas import (
    FlowListResponse, FlowResponse, TrafficAnalysisRequest, TrafficAnalysisResponse
)
from detection.supervised.predictor import ModelPredictor
from detection.anomaly.isolation_forest import AnomalyDetector
from mitigation.risk_engine import RiskEngine
from detection.explainability.shap_explainer import ShapExplainer
from backend.core.config import get_config

router = APIRouter()
predictor = ModelPredictor()
anomaly_detector = AnomalyDetector()
risk_engine = RiskEngine()
shap_explainer = ShapExplainer()


@router.get("/traffic", response_model=FlowListResponse)
def list_traffic(
    page: int = Query(1, ge=1),
    page_size: int = Query(50, ge=1, le=500),
    label: Optional[str] = None,
    protocol: Optional[str] = None,
    db: Session = Depends(get_db)
):
    query = db.query(NetworkFlow)
    if label:
        query = query.filter(NetworkFlow.label == label)
    if protocol:
        query = query.filter(NetworkFlow.protocol == protocol)

    total = query.count()
    flows = (
        query.order_by(NetworkFlow.created_at.desc())
        .offset((page - 1) * page_size)
        .limit(page_size)
        .all()
    )
    return FlowListResponse(
        total=total,
        page=page,
        page_size=page_size,
        flows=[FlowResponse.model_validate(f) for f in flows]
    )


@router.post("/traffic/analyse", response_model=TrafficAnalysisResponse)
def analyse_traffic(req: TrafficAnalysisRequest):
    features = req.features
    config = get_config()
    model_name = config.active_model

    # 1. Supervised prediction
    predicted_label = "benign"
    attack_type = "normal"
    confidence = 0.5
    shap_vals = None

    if model_name:
        try:
            pred_res = predictor.predict(model_name, features)
            predicted_label = str(pred_res.get("prediction", "benign"))
            confidence = float(pred_res.get("confidence", 0.5))
            attack_type = str(pred_res.get("predicted_class", predicted_label))
            
            # SHAP explanation
            try:
                shap_res = shap_explainer.explain_sample(model_name, features)
                if isinstance(shap_res, dict) and "feature_importance" in shap_res:
                    shap_vals = shap_res["feature_importance"]
                elif isinstance(shap_res, dict):
                    shap_vals = shap_res
            except Exception:
                shap_vals = None
        except Exception as e:
            # Fallback if model not loaded or mismatch
            pass

    # 2. Anomaly detection
    anomaly_score = 0.0
    try:
        anom_res = anomaly_detector.predict_single(features)
        anomaly_score = float(anom_res.get("anomaly_score", 0.0))
    except Exception:
        pass

    # 3. Risk scoring
    assessment = risk_engine.assess(
        model_confidence=confidence,
        attack_type=attack_type,
        anomaly_score=anomaly_score,
        frequency_score=10.0
    )

    return TrafficAnalysisResponse(
        predicted_label=predicted_label,
        attack_type=attack_type,
        confidence=confidence,
        anomaly_score=anomaly_score,
        risk_score=assessment.risk_score,
        severity=assessment.severity,
        shap_values=shap_vals
    )
