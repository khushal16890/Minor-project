# Installation & Deployment Guide

## 1. Prerequisites

- **Python**: 3.11 or higher
- **Operating Systems**:
  - **Linux (Ubuntu 22.04 / 24.04 LTS recommended)**: Full capabilities (Live interface capture, Suricata, Zeek, Linux nftables).
  - **Windows 10/11**: Fully functional in **Offline Mode** (Dataset imports, PCAP processing, ML training/evaluations, SHAP explainability, dashboard, and simulated mitigation).
- **Optional Tools**:
  - `tcpdump` / `libpcap-dev`
  - `suricata` (for live signature detection)
  - `zeek` (for connection metadata generation)
  - `nftables` (for live firewall blocking)

---

## 2. Local Setup (Virtual Environment)

### Step 1: Clone and Navigate
```bash
cd network-ids
```

### Step 2: Create and Activate Virtual Environment
On Linux / macOS:
```bash
python3 -m venv venv
source venv/bin/activate
```
On Windows PowerShell:
```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

### Step 3: Install Dependencies
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### Step 4: Environment Configuration
Copy the sample environment file:
```bash
cp .env.example .env
```

### Step 5: Seed Demo Data & Generate Sample Dataset
```bash
python scripts/generate_sample_data.py
```

---

## 3. Running the System

The system operates via two concurrent services:
1. **FastAPI Backend Service**: Serves the RESTful endpoints, database queries, and inference jobs.
2. **Streamlit Operations Dashboard**: Provides the interactive dark SOC interface.

### Running Backend API
```bash
uvicorn backend.main:app --host 127.0.0.1 --port 8000 --reload
```
API Documentation will be available at `http://127.0.0.1:8000/docs`.

### Running Frontend Dashboard
In a separate terminal:
```bash
streamlit run frontend/app.py --server.port 8501
```
Access the Operations Dashboard in your browser at `http://localhost:8501`.

---

## 4. Docker Deployment

To launch the entire containerized stack:
```bash
docker-compose up --build
```
- Backend API: `http://localhost:8000`
- Dashboard: `http://localhost:8501`

---

## 5. Linux-Specific Capabilities Setup

### Suricata Ingestion
1. Install Suricata:
   ```bash
   sudo apt install suricata -y
   ```
2. Enable EVE JSON logging in `/etc/suricata/suricata.yaml`:
   ```yaml
   outputs:
     - eve-log:
         enabled: yes
         type: file
         filename: eve.json
   ```
3. Set the log path in `.env`:
   ```env
   SURICATA_EVE_PATH=/var/log/suricata/eve.json
   ```

### Zeek Log Ingestion
1. Install Zeek:
   ```bash
   sudo apt install zeek -y
   ```
2. Point Zeek logs directory in `.env`:
   ```env
   ZEEK_LOG_DIR=/opt/zeek/logs/current
   ```

### Live nftables Firewall Mitigation
1. Ensure the user running the process has `sudo` access or `CAP_NET_ADMIN` capability.
2. Enable live mitigation in `.env`:
   ```env
   MITIGATION_ENABLED=true
   MITIGATION_SIMULATION=false
   ```
