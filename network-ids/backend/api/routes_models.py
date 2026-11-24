"""
ML model management, training, and evaluation API router.
"""
from typing import List, Optional
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
import pandas as pd
from backend.database.models import get_db, ModelInfo, ModelEvaluation
from backend.schemas.schemas import (
    ModelInfoResponse, ModelEvalResponse, TrainRequest, EvaluateRequest
)
from backend.core.config import get_config, update_config, RAW_DATA_DIR, PROCESSED_DATA_DIR
from detection.supervised.trainer import ModelTrainer
from detection.supervised.evaluator import ModelEvaluator
from detection.preprocessing.dataset_loader import DatasetLoader

router = APIRouter()
trainer = ModelTrainer()
evaluator = ModelEvaluator()
loader = DatasetLoader()


@router.get("/models", response_model=List[ModelInfoResponse])
def list_models(db: Session = Depends(get_db)):
    models = db.query(ModelInfo).order_by(ModelInfo.created_at.desc()).all()
    return [ModelInfoResponse.model_validate(m) for m in models]


@router.post("/models/train", response_model=ModelInfoResponse)
def train_model(req: TrainRequest, db: Session = Depends(get_db)):
    # Find dataset file in processed or raw
    csv_path = None
    for p in [PROCESSED_DATA_DIR / f"{req.dataset_name}.csv", RAW_DATA_DIR / f"{req.dataset_name}.csv", PROCESSED_DATA_DIR / req.dataset_name, RAW_DATA_DIR / req.dataset_name]:
        if p.exists() and p.is_file():
            csv_path = p
            break

    if not csv_path:
        raise HTTPException(status_code=404, detail=f"Dataset {req.dataset_name} not found")

    df = pd.read_csv(csv_path)
    # Check label column
    label_col = None
    for candidate in ["Label", "label", "attack_cat", "class", "target"]:
        if candidate in df.columns:
            label_col = candidate
            break

    if not label_col:
        raise HTTPException(status_code=400, detail="Dataset does not contain a recognizable label column")

    y = df[label_col]
    X = df.drop(columns=[label_col])
    # Drop identifier columns if present
    for drop_col in ["id", "Flow ID", "flow_id", "src_ip", "dst_ip", "Timestamp", "timestamp"]:
        if drop_col in X.columns:
            X = X.drop(columns=[drop_col])

    # Convert binary if task is binary
    if req.task == "binary":
        y = y.apply(lambda v: 0 if str(v).strip().lower() in ["0", "benign", "normal"] else 1)

    model_name = f"{req.model_type}_{req.task}_{req.dataset_name}"
    try:
        train_res = trainer.train(
            X, y,
            model_name=model_name,
            dataset_name=req.dataset_name,
            apply_smote=True,
            save=True
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Training failed: {str(e)}")

    # Update or insert ModelInfo in DB
    model_record = db.query(ModelInfo).filter(ModelInfo.name == model_name).first()
    if not model_record:
        model_record = ModelInfo(
            name=model_name,
            model_type=req.model_type,
            task=req.task,
            dataset=req.dataset_name,
            file_path=train_res.get("model_path"),
            scaler_path=train_res.get("scaler_path"),
            feature_columns=train_res.get("feature_columns"),
            hyperparameters=req.hyperparameters or {},
            training_samples=len(X),
            is_active=True
        )
        db.add(model_record)
    else:
        model_record.file_path = train_res.get("model_path")
        model_record.scaler_path = train_res.get("scaler_path")
        model_record.feature_columns = train_res.get("feature_columns")
        model_record.training_samples = len(X)
        model_record.is_active = True

    # Deactivate others if this is set active
    db.query(ModelInfo).filter(ModelInfo.name != model_name).update({"is_active": False})
    db.commit()
    db.refresh(model_record)

    # Set as active in config
    cfg = get_config()
    cfg.active_model = model_name
    update_config(cfg)

    return ModelInfoResponse.model_validate(model_record)


@router.post("/models/evaluate", response_model=ModelEvalResponse)
def evaluate_model(req: EvaluateRequest, db: Session = Depends(get_db)):
    model_record = db.query(ModelInfo).filter(ModelInfo.name == req.model_name).first()
    if not model_record:
        raise HTTPException(status_code=404, detail=f"Model {req.model_name} not found")

    csv_path = None
    for p in [PROCESSED_DATA_DIR / f"{req.dataset_name}.csv", RAW_DATA_DIR / f"{req.dataset_name}.csv", PROCESSED_DATA_DIR / req.dataset_name, RAW_DATA_DIR / req.dataset_name]:
        if p.exists() and p.is_file():
            csv_path = p
            break

    if not csv_path:
        raise HTTPException(status_code=404, detail=f"Dataset {req.dataset_name} not found")

    df = pd.read_csv(csv_path)
    label_col = None
    for candidate in ["Label", "label", "attack_cat", "class", "target"]:
        if candidate in df.columns:
            label_col = candidate
            break

    if not label_col:
        raise HTTPException(status_code=400, detail="Dataset does not contain a recognizable label column")

    y = df[label_col]
    X = df.drop(columns=[label_col])
    for drop_col in ["id", "Flow ID", "flow_id", "src_ip", "dst_ip", "Timestamp", "timestamp"]:
        if drop_col in X.columns:
            X = X.drop(columns=[drop_col])

    if model_record.task == "binary":
        y = y.apply(lambda v: 0 if str(v).strip().lower() in ["0", "benign", "normal"] else 1)

    try:
        eval_res = evaluator.evaluate(
            model_name=req.model_name,
            X_test=X,
            y_test=y,
            dataset_name=req.dataset_name
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Evaluation failed: {str(e)}")

    eval_record = ModelEvaluation(
        model_id=model_record.id,
        dataset=req.dataset_name,
        task=model_record.task,
        accuracy=eval_res.get("accuracy"),
        precision_macro=eval_res.get("precision_macro"),
        recall_macro=eval_res.get("recall_macro"),
        f1_macro=eval_res.get("f1_macro"),
        f1_weighted=eval_res.get("f1_weighted"),
        false_positive_rate=eval_res.get("false_positive_rate"),
        inference_latency_ms=eval_res.get("inference_latency_ms"),
        confusion_matrix=eval_res.get("confusion_matrix"),
        classification_report=eval_res.get("classification_report"),
    )
    db.add(eval_record)
    db.commit()
    db.refresh(eval_record)

    return ModelEvalResponse.model_validate(eval_record)


@router.post("/models/{model_name}/activate")
def activate_model(model_name: str, db: Session = Depends(get_db)):
    model_record = db.query(ModelInfo).filter(ModelInfo.name == model_name).first()
    if not model_record:
        raise HTTPException(status_code=404, detail="Model not found")

    db.query(ModelInfo).update({"is_active": False})
    model_record.is_active = True
    db.commit()

    cfg = get_config()
    cfg.active_model = model_name
    update_config(cfg)

    return {"message": f"Model {model_name} is now active"}
