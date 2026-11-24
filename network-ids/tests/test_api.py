"""
Integration tests for FastAPI REST API endpoints.
"""
import pytest
from fastapi.testclient import TestClient
from backend.main import app


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c


def test_api_health(client):
    response = client.get("/api/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert "version" in data


def test_api_dashboard_overview(client):
    response = client.get("/api/dashboard/overview")
    assert response.status_code == 200
    data = response.json()
    assert "total_flows" in data
    assert "active_incidents" in data
    assert "active_blocks" in data


def test_api_traffic_list(client):
    response = client.get("/api/traffic?page=1&page_size=10")
    assert response.status_code == 200
    data = response.json()
    assert "flows" in data
    assert "total" in data


def test_api_incidents_list(client):
    response = client.get("/api/incidents?page=1&page_size=10")
    assert response.status_code == 200
    data = response.json()
    assert "incidents" in data
    assert "total" in data


def test_api_settings_get_and_post(client):
    get_res = client.get("/api/settings")
    assert get_res.status_code == 200
    cfg = get_res.json()
    assert "risk_weights" in cfg

    # Update
    post_res = client.post("/api/settings", json={"risk_weights": {"model_confidence": 0.50}})
    assert post_res.status_code == 200
    updated_cfg = post_res.json()
    assert updated_cfg["risk_weights"]["model_confidence"] == 0.50
