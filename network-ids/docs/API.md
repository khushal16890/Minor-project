# REST API Reference Documentation

The FastAPI backend exposes RESTful endpoints at `http://127.0.0.1:8000/api` (Swagger UI available at `/docs`).

---

## 1. Health & Status

### `GET /api/health`
Returns system status, active database connection, number of loaded models, and uptime.

**Response:**
```json
{
  "status": "ok",
  "version": "1.0.0",
  "database": "connected",
  "models_loaded": 4,
  "uptime_seconds": 320.5
}
```

---

## 2. Security Operations Dashboard

### `GET /api/dashboard/overview`
Retrieves aggregated telemetry for the Security Overview dashboard page.

**Response:**
```json
{
  "total_flows": 560,
  "benign_flows": 380,
  "malicious_flows": 180,
  "active_incidents": 24,
  "critical_incidents": 6,
  "active_blocks": 2,
  "recent_events": [ ... ],
  "attack_distribution": { "DoS": 85, "PortScan": 45, "Brute Force": 30 },
  "detection_method_counts": { "hybrid": 12, "ml": 8, "signature": 4 }
}
```

---

## 3. Network Traffic & Flow Analysis

### `GET /api/traffic`
Retrieves paginated network flows.
- **Query Params**: `page` (default 1), `page_size` (default 50), `protocol` (e.g. `tcp`), `label` (e.g. `BENIGN`)

### `POST /api/traffic/analyse`
Performs real-time multi-engine analysis on a single flow feature vector.

**Request:**
```json
{
  "features": {
    "flow_duration": 0.45,
    "protocol_type": "tcp",
    "src_port": 54210,
    "dst_port": 80,
    "fwd_packets": 250,
    "bwd_packets": 2,
    "total_fwd_bytes": 14000,
    "total_bwd_bytes": 120,
    "flow_pkts_per_sec": 555.5
  }
}
```

**Response:**
```json
{
  "predicted_label": "DoS",
  "attack_type": "dos",
  "confidence": 0.94,
  "anomaly_score": 82.5,
  "risk_score": 86.2,
  "severity": "critical",
  "shap_values": {
    "flow_pkts_per_sec": 0.42,
    "fwd_packets": 0.31,
    "flow_duration": -0.12
  }
}
```

---

## 4. Incidents & Forensics

### `GET /api/incidents`
Lists detected security incidents.
- **Query Params**: `page`, `page_size`, `severity` (`low`, `medium`, `high`, `critical`), `detection_method`, `status`

### `GET /api/incidents/{incident_id}`
Returns granular telemetry, SHAP values, and risk evidence for a specific incident.

### `POST /api/incidents/{incident_id}/resolve`
Marks an incident as resolved or false positive.

---

## 5. Machine Learning Models

### `GET /api/models`
Lists all trained models registered in the database and models directory.

### `POST /api/models/train`
Triggers training of a candidate supervised classifier.
```json
{
  "model_type": "xgboost",
  "task": "binary",
  "dataset_name": "sample_traffic",
  "test_size": 0.2,
  "hyperparameters": { "n_estimators": 100, "max_depth": 6 }
}
```

### `POST /api/models/evaluate`
Evaluates a model against a dataset and returns confusion matrix, macro-F1, FPR, and latency.

### `POST /api/models/{model_name}/activate`
Sets the active model used for live inference.

---

## 6. Datasets & PCAP

### `GET /api/datasets`
Lists available raw and processed CSV datasets.

### `POST /api/datasets/upload`
Uploads a new CSV dataset file (`multipart/form-data`).

### `POST /api/datasets/validate?dataset_name={name}`
Validates schema compatibility, missing values, duplicates, and class distributions.

### `POST /api/pcap/analyse`
Uploads a PCAP file (`multipart/form-data`), extracts flows, evaluates each flow against detection engines, and logs incidents.

---

## 7. Mitigation & Firewall Control

### `GET /api/mitigation/active`
Lists active IP blocks and expiry times.

### `GET /api/mitigation/history`
Returns immutable mitigation history and audit records.

### `POST /api/mitigation/block`
```json
{
  "target_ip": "198.51.100.45",
  "reason": "Repeated DoS Flood",
  "duration_minutes": 60,
  "incident_id": "optional-incident-uuid"
}
```

### `POST /api/mitigation/unblock`
```json
{
  "target_ip": "198.51.100.45",
  "reason": "Admin manual remediation"
}
```

---

## 8. System Configuration

### `GET /api/settings`
Retrieves current risk weights, thresholds, detection parameters, and mitigation modes.

### `POST /api/settings`
Updates runtime configuration dynamically.
