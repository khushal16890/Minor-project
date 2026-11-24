"""
Main FastAPI Application Entry Point.
Network Intrusion Detection & Prevention System.
"""
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from backend.database.models import init_db
from backend.api import (
    routes_health,
    routes_dashboard,
    routes_traffic,
    routes_incidents,
    routes_models,
    routes_datasets,
    routes_pcap,
    routes_mitigation,
    routes_settings,
)


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: ensure tables exist
    init_db()
    yield
    # Shutdown logic if any


app = FastAPI(
    title="Hybrid Network IDS/IPS Backend API",
    description="Intelligent Hybrid Network Intrusion Detection, Analysis and Automated Prevention System API",
    version="1.0.0",
    lifespan=lifespan,
)

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include all routers under /api
app.include_router(routes_health.router, prefix="/api", tags=["Health"])
app.include_router(routes_dashboard.router, prefix="/api", tags=["Dashboard"])
app.include_router(routes_traffic.router, prefix="/api", tags=["Traffic"])
app.include_router(routes_incidents.router, prefix="/api", tags=["Incidents"])
app.include_router(routes_models.router, prefix="/api", tags=["Models"])
app.include_router(routes_datasets.router, prefix="/api", tags=["Datasets"])
app.include_router(routes_pcap.router, prefix="/api", tags=["PCAP"])
app.include_router(routes_mitigation.router, prefix="/api", tags=["Mitigation"])
app.include_router(routes_settings.router, prefix="/api", tags=["Settings"])


@app.get("/")
def root():
    return {
        "system": "Intelligent Hybrid Network IDS/IPS",
        "status": "operational",
        "api_docs": "/docs",
        "health": "/api/health",
    }


if __name__ == "__main__":
    import uvicorn
    from backend.core.config import API_HOST, API_PORT
    uvicorn.run("backend.main:app", host=API_HOST, port=API_PORT, reload=True)
