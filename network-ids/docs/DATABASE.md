# Database Schema & Entity Documentation

The system utilizes SQLite (compatible with PostgreSQL/MySQL via SQLAlchemy) with tables optimized for time-series flow indexing and incident auditability.

---

## 1. Entity-Relationship Diagram

```mermaid
erDiagram
    NETWORK_FLOWS ||--o{ DETECTION_INCIDENTS : generates
    DETECTION_INCIDENTS ||--o{ MITIGATION_ACTIONS : triggers
    MODEL_INFO ||--o{ MODEL_EVALUATIONS : records

    NETWORK_FLOWS {
        string id PK
        string flow_id
        datetime timestamp
        string src_ip
        string dst_ip
        int src_port
        int dst_port
        string protocol
        float duration
        int total_bytes
        string label
        json features
    }

    DETECTION_INCIDENTS {
        string id PK
        string flow_id FK
        datetime timestamp
        string detection_method
        string attack_type
        float model_confidence
        float anomaly_score
        float risk_score
        string severity
        json shap_values
        string status
    }

    MODEL_INFO {
        string id PK
        string name
        string model_type
        string task
        string file_path
        boolean is_active
    }

    MODEL_EVALUATIONS {
        string id PK
        string model_id FK
        float accuracy
        float f1_macro
        float false_positive_rate
        float inference_latency_ms
        json confusion_matrix
    }

    MITIGATION_ACTIONS {
        string id PK
        string action_type
        string target_ip
        string reason
        datetime expires_at
        boolean is_active
        boolean is_simulated
    }

    AUDIT_LOGS {
        string id PK
        datetime timestamp
        string action
        string module
        json details
    }
```

---

## 2. Table Specifications

### 2.1 `network_flows`
Stores individual network flows extracted from PCAPs, datasets, or live captures.
- **Indexes**: `ix_flow_src_dst` on `(src_ip, dst_ip)`, `ix_flow_label` on `label`.

### 2.2 `detection_incidents`
Stores security events flagged by signature, ML, anomaly, or hybrid engines.
- **Indexes**: `ix_incident_severity`, `ix_incident_status`, `ix_incident_method`.

### 2.3 `model_info` & `model_evaluations`
Maintains model artifact metadata, trained version history, and empirical benchmark evaluation results.

### 2.4 `mitigation_actions`
Maintains the state of all firewall containment actions (active, expired, unblocked) and simulation status.

### 2.5 `audit_logs`
Append-only log recording critical administrative events, model deployments, and firewall changes.
