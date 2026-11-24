"""
Dataset upload and validation API router.
"""
from pathlib import Path
import os
import shutil
from typing import List, Dict, Any
from fastapi import APIRouter, UploadFile, File, HTTPException
import pandas as pd
from backend.core.config import RAW_DATA_DIR, PROCESSED_DATA_DIR
from backend.schemas.schemas import DatasetUploadResponse, DatasetValidationResponse
from detection.preprocessing.dataset_loader import DatasetLoader

router = APIRouter()
loader = DatasetLoader()


@router.get("/datasets", response_model=List[Dict[str, Any]])
def list_datasets():
    datasets = []
    for d in [RAW_DATA_DIR, PROCESSED_DATA_DIR]:
        if d.exists():
            for f in d.glob("*.csv"):
                datasets.append({
                    "name": f.stem,
                    "filename": f.name,
                    "type": "processed" if d == PROCESSED_DATA_DIR else "raw",
                    "size_bytes": f.stat().st_size,
                    "path": str(f)
                })
    return datasets


@router.post("/datasets/upload", response_model=DatasetUploadResponse)
async def upload_dataset(file: UploadFile = File(...)):
    if not file.filename.endswith(".csv"):
        raise HTTPException(status_code=400, detail="Only CSV files are supported")

    dest_path = RAW_DATA_DIR / file.filename
    try:
        with open(dest_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to save file: {str(e)}")

    try:
        # Read a preview or line count
        df = pd.read_csv(dest_path, nrows=5)
        cols = len(df.columns)
        # Count total rows
        with open(dest_path, "r", encoding="utf-8", errors="ignore") as f:
            total_rows = sum(1 for _ in f) - 1
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid CSV file: {str(e)}")

    return DatasetUploadResponse(
        name=file.filename,
        file_path=str(dest_path),
        rows=max(0, total_rows),
        columns=cols,
        message="Dataset uploaded successfully"
    )


@router.post("/datasets/validate", response_model=DatasetValidationResponse)
def validate_dataset(dataset_name: str):
    csv_path = None
    for p in [RAW_DATA_DIR / f"{dataset_name}.csv", RAW_DATA_DIR / dataset_name, PROCESSED_DATA_DIR / f"{dataset_name}.csv", PROCESSED_DATA_DIR / dataset_name]:
        if p.exists() and p.is_file():
            csv_path = p
            break

    if not csv_path:
        raise HTTPException(status_code=404, detail=f"Dataset {dataset_name} not found")

    try:
        # Read dataset (sample if very large to prevent memory issues)
        df = pd.read_csv(csv_path, nrows=50000)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to read dataset: {str(e)}")

    rows, cols = df.shape
    col_names = list(df.columns)
    missing = df.isnull().sum().to_dict()
    missing_report = {k: int(v) for k, v in missing.items() if v > 0}
    dup_count = int(df.duplicated().sum())

    # Class distribution
    class_dist = {}
    for candidate in ["Label", "label", "attack_cat", "class", "target"]:
        if candidate in df.columns:
            class_dist = {str(k): int(v) for k, v in df[candidate].value_counts().items()}
            break

    warnings = []
    if dup_count > 0:
        warnings.append(f"Found {dup_count} duplicate rows.")
    if sum(missing.values()) > 0:
        warnings.append(f"Found {sum(missing.values())} missing values across columns.")

    return DatasetValidationResponse(
        name=dataset_name,
        rows=rows,
        columns=cols,
        column_names=col_names,
        missing_values=missing_report,
        duplicate_rows=dup_count,
        class_distribution=class_dist,
        feature_mapping_status="Mapped" if class_dist else "Unmapped",
        warnings=warnings,
        preprocessing_summary={
            "numeric_columns": len(df.select_dtypes(include=["number"]).columns),
            "categorical_columns": len(df.select_dtypes(include=["object"]).columns)
        }
    )
