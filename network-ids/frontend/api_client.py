"""
API Client for communicating with the FastAPI backend.
Handles all HTTP requests with robust error handling and timeouts.
"""
import os
import httpx
from typing import Optional, Dict, Any, List

API_BASE_URL = os.getenv("API_BASE_URL", "http://127.0.0.1:8000/api")


class APIClient:
    def __init__(self, base_url: str = API_BASE_URL, timeout: float = 15.0):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    def _get(self, endpoint: str, params: Optional[Dict[str, Any]] = None) -> Optional[Any]:
        try:
            with httpx.Client(timeout=self.timeout) as client:
                r = client.get(f"{self.base_url}{endpoint}", params=params)
                if r.status_code == 200:
                    return r.json()
                return None
        except Exception:
            return None

    def _post(self, endpoint: str, json_data: Optional[Dict[str, Any]] = None, files: Optional[Dict[str, Any]] = None, params: Optional[Dict[str, Any]] = None) -> Optional[Any]:
        try:
            with httpx.Client(timeout=self.timeout) as client:
                r = client.post(f"{self.base_url}{endpoint}", json=json_data, files=files, params=params)
                if r.status_code in (200, 201):
                    return r.json()
                return None
        except Exception:
            return None

    # Health
    def get_health(self) -> Optional[Dict[str, Any]]:
        return self._get("/health")

    # Dashboard Overview
    def get_dashboard_overview(self) -> Optional[Dict[str, Any]]:
        return self._get("/dashboard/overview")

    # Traffic
    def get_traffic(self, page: int = 1, page_size: int = 50, label: Optional[str] = None, protocol: Optional[str] = None) -> Optional[Dict[str, Any]]:
        params = {"page": page, "page_size": page_size}
        if label:
            params["label"] = label
        if protocol:
            params["protocol"] = protocol
        return self._get("/traffic", params=params)

    def analyse_traffic(self, features: Dict[str, float]) -> Optional[Dict[str, Any]]:
        return self._post("/traffic/analyse", json_data={"features": features})

    # Incidents
    def get_incidents(self, page: int = 1, page_size: int = 50, severity: Optional[str] = None, detection_method: Optional[str] = None, status: Optional[str] = None) -> Optional[Dict[str, Any]]:
        params = {"page": page, "page_size": page_size}
        if severity:
            params["severity"] = severity
        if detection_method:
            params["detection_method"] = detection_method
        if status:
            params["status"] = status
        return self._get("/incidents", params=params)

    def get_incident(self, incident_id: str) -> Optional[Dict[str, Any]]:
        return self._get(f"/incidents/{incident_id}")

    def resolve_incident(self, incident_id: str, status: str = "resolved", notes: Optional[str] = None) -> Optional[Dict[str, Any]]:
        params = {"status": status}
        if notes:
            params["notes"] = notes
        return self._post(f"/incidents/{incident_id}/resolve", params=params)

    # Models
    def get_models(self) -> Optional[List[Dict[str, Any]]]:
        return self._get("/models")

    def train_model(self, model_type: str, task: str, dataset_name: str, hyperparameters: Optional[Dict[str, Any]] = None) -> Optional[Dict[str, Any]]:
        data = {
            "model_type": model_type,
            "task": task,
            "dataset_name": dataset_name,
            "hyperparameters": hyperparameters or {}
        }
        return self._post("/models/train", json_data=data)

    def evaluate_model(self, model_name: str, dataset_name: str) -> Optional[Dict[str, Any]]:
        data = {"model_name": model_name, "dataset_name": dataset_name}
        return self._post("/models/evaluate", json_data=data)

    def activate_model(self, model_name: str) -> Optional[Dict[str, Any]]:
        return self._post(f"/models/{model_name}/activate")

    # Datasets
    def get_datasets(self) -> Optional[List[Dict[str, Any]]]:
        return self._get("/datasets")

    def upload_dataset(self, filename: str, content: bytes) -> Optional[Dict[str, Any]]:
        files = {"file": (filename, content, "text/csv")}
        return self._post("/datasets/upload", files=files)

    def validate_dataset(self, dataset_name: str) -> Optional[Dict[str, Any]]:
        return self._post("/datasets/validate", params={"dataset_name": dataset_name})

    # PCAP
    def analyse_pcap(self, filename: str, content: bytes) -> Optional[Dict[str, Any]]:
        files = {"file": (filename, content, "application/octet-stream")}
        return self._post("/pcap/analyse", files=files)

    # Mitigation
    def get_active_blocks(self) -> Optional[Dict[str, Any]]:
        return self._get("/mitigation/active")

    def get_mitigation_history(self) -> Optional[Dict[str, Any]]:
        return self._get("/mitigation/history")

    def block_ip(self, target_ip: str, reason: str, duration_minutes: int = 60, incident_id: Optional[str] = None) -> Optional[Dict[str, Any]]:
        data = {
            "target_ip": target_ip,
            "reason": reason,
            "duration_minutes": duration_minutes,
            "incident_id": incident_id
        }
        return self._post("/mitigation/block", json_data=data)

    def unblock_ip(self, target_ip: str, reason: Optional[str] = None) -> Optional[Dict[str, Any]]:
        data = {"target_ip": target_ip, "reason": reason or "Admin manual unblock"}
        return self._post("/mitigation/unblock", json_data=data)

    # Settings
    def get_settings(self) -> Optional[Dict[str, Any]]:
        return self._get("/settings")

    def update_settings(self, settings_update: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        return self._post("/settings", json_data=settings_update)
