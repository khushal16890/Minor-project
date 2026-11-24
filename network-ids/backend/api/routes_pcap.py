"""
PCAP processing and offline flow analysis API router.
"""
from pathlib import Path
import shutil
import tempfile
from fastapi import APIRouter, UploadFile, File, HTTPException, Depends
from sqlalchemy.orm import Session
from backend.database.models import get_db, NetworkFlow, DetectionIncident
from backend.schemas.schemas import PcapAnalysisResponse
from network.pcap.pcap_processor import PcapProcessor
from detection.supervised.predictor import ModelPredictor
from detection.anomaly.isolation_forest import AnomalyDetector
from mitigation.risk_engine import RiskEngine
from backend.core.config import get_config

router = APIRouter()
processor = PcapProcessor()
predictor = ModelPredictor()
anomaly_detector = AnomalyDetector()
risk_engine = RiskEngine()


@router.post("/pcap/analyse", response_model=PcapAnalysisResponse)
async def analyse_pcap(file: UploadFile = File(...), db: Session = Depends(get_db)):
    if not (file.filename.endswith(".pcap") or file.filename.endswith(".pcapng")):
        raise HTTPException(status_code=400, detail="Only PCAP / PCAPNG files are supported")

    with tempfile.NamedTemporaryFile(delete=False, suffix=".pcap") as tmp:
        shutil.copyfileobj(file.file, tmp)
        tmp_path = tmp.name

    try:
        flows = processor.process_file(tmp_path)
    except Exception as e:
        Path(tmp_path).unlink(missing_ok=True)
        raise HTTPException(status_code=500, detail=f"Failed to process PCAP: {str(e)}")

    Path(tmp_path).unlink(missing_ok=True)

    config = get_config()
    model_name = config.active_model
    incidents_count = 0
    total_pkts = sum(f.get("fwd_packets", 0) + f.get("bwd_packets", 0) for f in flows)

    for flow in flows:
        # Save flow to DB
        db_flow = NetworkFlow(
            flow_id=flow.get("flow_id"),
            src_ip=flow.get("src_ip"),
            dst_ip=flow.get("dst_ip"),
            src_port=flow.get("src_port"),
            dst_port=flow.get("dst_port"),
            protocol=str(flow.get("protocol_type", "tcp")),
            duration=flow.get("flow_duration", 0.0),
            total_fwd_packets=flow.get("fwd_packets", 0),
            total_bwd_packets=flow.get("bwd_packets", 0),
            total_bytes=flow.get("total_fwd_bytes", 0) + flow.get("total_bwd_bytes", 0),
            source="pcap",
            features=flow
        )
        db.add(db_flow)
        db.flush()

        # Classify if model available
        pred_label = "benign"
        confidence = 0.5
        attack_type = "normal"
        if model_name:
            try:
                pred = predictor.predict(model_name, flow)
                pred_label = str(pred.get("prediction", "benign"))
                confidence = float(pred.get("confidence", 0.5))
                attack_type = str(pred.get("predicted_class", pred_label))
            except Exception:
                pass

        # Anomaly score
        anom_score = 0.0
        try:
            anom = anomaly_detector.predict_single(flow)
            anom_score = float(anom.get("anomaly_score", 0.0))
        except Exception:
            pass

        # Risk assessment
        assessment = risk_engine.assess(
            model_confidence=confidence,
            attack_type=attack_type,
            anomaly_score=anom_score,
            frequency_score=5.0
        )

        # Flag incident if malicious or high anomaly
        if pred_label.lower() not in ["benign", "0", "normal"] or assessment.severity in ["high", "critical"]:
            incident = DetectionIncident(
                flow_id=db_flow.id,
                detection_method="ml" if pred_label.lower() not in ["benign", "0", "normal"] else "anomaly",
                predicted_label=pred_label,
                attack_type=attack_type,
                model_confidence=confidence,
                anomaly_score=anom_score,
                risk_score=assessment.risk_score,
                severity=assessment.severity,
                src_ip=flow.get("src_ip"),
                dst_ip=flow.get("dst_ip"),
                src_port=flow.get("src_port"),
                dst_port=flow.get("dst_port"),
                protocol=str(flow.get("protocol_type", "tcp")),
                evidence={"risk_factors": assessment.risk_factors},
                status="open"
            )
            db.add(incident)
            incidents_count += 1

    db.commit()

    return PcapAnalysisResponse(
        file_name=file.filename,
        total_packets=total_pkts,
        flows_extracted=len(flows),
        incidents_detected=incidents_count,
        summary={
            "flows_count": len(flows),
            "incidents_count": incidents_count,
            "sample_flows": flows[:3] if flows else []
        }
    )
