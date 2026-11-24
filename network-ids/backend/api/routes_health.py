"""
Health check endpoints.
"""
import time
from fastapi import APIRouter
from backend.schemas.schemas import HealthResponse
from backend.database.models import SessionLocal
from backend.core.config import MODELS_DIR

router = APIRouter()
START_TIME = time.time()


@router.get("/health", response_model=HealthResponse)
def health_check():
    db_status = "connected"
    try:
        db = SessionLocal()
        db.execute("SELECT 1")
        db.close()
    except Exception:
        db_status = "error"

    models_count = 0
    if MODELS_DIR.exists():
        models_count = len(list(MODELS_DIR.glob("*.joblib")))

    return HealthResponse(
        status="ok",
        version="1.0.0",
        database=db_status,
        models_loaded=models_count,
        uptime_seconds=round(time.time() - START_TIME, 2),
    )
