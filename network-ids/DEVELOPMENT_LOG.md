# Development Log

## Engineering Timeline & Milestones

### Phase 1: Environment & Foundational Architecture
- Inspected execution environment (Python 3.11.4 on Windows x86_64).
- Verified installed ML dependencies: `xgboost==3.1.2`, `lightgbm==4.6.0`, `scikit-learn==1.4.2`, `shap==0.50.0`, `imbalanced-learn==0.13.0`, `sqlalchemy==2.0.37`.
- Installed necessary missing packages: `fastapi==0.165.0`, `uvicorn==0.54.0`, `python-multipart`, `aiofiles`, `scapy==2.8.0`, `pytest==9.2.0`, `pytest-asyncio`.
- Designed directory structure adhering to separation of concerns across `backend`, `detection`, `network`, `mitigation`, `frontend`, `datasets`, `models`, `experiments`, `tests`, `scripts`, and `docs`.
- Established master configuration (`backend/core/config.py`) including risk scoring weights, attack severity mappings, and the internal 65-feature schema (`FEATURE_SCHEMA`).
- Implemented relational database schema in SQLite via SQLAlchemy (`backend/database/models.py`) with tables for flows, incidents, model metadata, evaluations, anomalies, alerts, mitigation actions, and audit logs.
- Defined Pydantic data validation schemas (`backend/schemas/schemas.py`).

### Phase 2: Feature Engineering & Detection Modules
- Implemented `PreprocessingPipeline` (`detection/preprocessing/pipeline.py`):
  - Robust handling of infinite values, NaNs, duplicate records.
  - Automatic exclusion of IP addresses, timestamps, and Flow IDs to prevent data leakage.
  - StandardScaler persistence with Joblib for identical train/inference transformations.
- Implemented `DatasetLoader` (`detection/preprocessing/dataset_loader.py`):
  - Column alignment and schema mapping for CIC-IDS2017 and UNSW-NB15 datasets.
- Implemented `FeatureExtractor` (`detection/feature_extraction/extractor.py`):
  - Extraction from raw packets, 5-tuples, Zeek logs, and PCAP metadata.
- Implemented Supervised ML Engine (`detection/supervised/`):
  - `ModelTrainer`: LogisticRegression, RandomForest, XGBoost, LightGBM with stratified train-test splits and training-only SMOTE oversampling.
  - `ModelPredictor`: In-memory model caching, probability extraction, latency benchmarking.
  - `ModelEvaluator`: Confusion matrix generation, precision/recall/macro-F1, false positive rate.
- Implemented Anomaly Detection Engine (`detection/anomaly/isolation_forest.py`):
  - Isolation Forest trained exclusively on benign baseline traffic, normalized 0–100 anomaly scoring.
- Implemented Network Ingestion Modules:
  - `SuricataParser`: EVE JSON parser, signature match extractor.
  - `ZeekParser`: Connection, DNS, and HTTP log reader.
  - `PcapProcessor`: Wireshark PCAP reader extracting network flows.
- Implemented Hybrid Correlation & Risk Scoring:
  - `HybridDetectionEngine`: Correlates signature alerts, ML probabilities, and anomaly scores with weighted scoring.
  - `ShapExplainer`: Local and global TreeExplainer feature attributions.
  - `RiskEngine`: Formula: $0.40 \times \text{Confidence} + 0.30 \times \text{Severity} + 0.20 \times \text{Anomaly} + 0.10 \times \text{Frequency}$.
  - `NftablesManager`: Linux firewall management with safe simulation mode, duplicate prevention, and protected IP guards.

### Phase 3: REST Backend & Frontend Operations Dashboard
- Built FastAPI backend service (`backend/main.py`) with 14 modular endpoints:
  - `/api/health`, `/api/dashboard/overview`, `/api/traffic`, `/api/incidents`, `/api/models`, `/api/datasets`, `/api/pcap`, `/api/mitigation`, `/api/settings`.
- Built Streamlit operations dashboard (`frontend/app.py`):
  - Dark SOC UI layout with responsive CSS styling.
  - PAGE 1: Security Overview (KPI metrics, attack categories, detection engines, recent events).
  - PAGE 2: Live Network Monitoring (Traffic volume, protocol distribution, flow telemetry).
  - PAGE 3: Threat Detection (Multi-engine alert stream, severity charts, triage).
  - PAGE 4: Incident Investigation (Detailed 5-tuple forensic view, SHAP feature attribution bar charts, risk breakdown, one-click block).
  - PAGE 5: Model Laboratory (Dataset import, schema validation, multi-model training, confusion matrix heatmaps, model registry).
  - PAGE 6: Mitigation Centre (Active blocked IPs table, manual block/unblock, duration management, audit log history).
  - PAGE 7: System Settings (Configurable risk weights sliders, mitigation policy toggles, detection thresholds).

### Phase 4: Verification, Experiments, and Quality Assurance
- Created sample synthetic generator (`scripts/generate_sample_data.py`) creating 500 network flows and seeding SQLite database.
- Executed Pytest test suite (`tests/`):
  - 18 unit and integration tests covering preprocessing, feature extraction, ML training/inference, Isolation Forest anomaly scoring, risk assessment, nftables simulation, and REST API endpoints.
  - Result: 18 passed, 0 failed (100% pass rate).
- Executed Experimental Evaluation Suite (`experiments/experiment_runner.py`):
  - Successfully executed all 8 experiments (E1 to E8) and exported `experiment_report_summary.json` and `experiment_report_summary.csv`.
- Generated Dockerfile and Docker Compose configurations for containerized deployment.
