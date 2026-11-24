"""
Synthetic Dataset & Demo Data Generator.
Creates realistic sample datasets matching CIC-IDS2017 & UNSW-NB15 schemas
for offline training, unit testing, and dashboard demonstration.
"""
import random
import datetime
import numpy as np
import pandas as pd
from pathlib import Path
from backend.core.config import RAW_DATA_DIR, FEATURE_SCHEMA
from backend.database.models import init_db, SessionLocal, NetworkFlow, DetectionIncident, MitigationAction, utcnow


def generate_synthetic_dataset(n_samples: int = 500, filename: str = "sample_traffic.csv") -> Path:
    RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
    out_path = RAW_DATA_DIR / filename

    protocols = ["tcp", "udp", "icmp"]
    labels = ["BENIGN", "DoS", "PortScan", "Brute Force", "Botnet"]
    label_weights = [0.60, 0.15, 0.10, 0.10, 0.05]

    data = []
    base_time = datetime.datetime.now() - datetime.timedelta(hours=2)

    for i in range(n_samples):
        lbl = random.choices(labels, weights=label_weights)[0]
        is_attack = lbl != "BENIGN"
        proto = random.choice(protocols)

        # Attack-specific characteristics
        if lbl == "DoS":
            duration = random.uniform(0.01, 0.5)
            fwd_pkts = random.randint(50, 1000)
            bwd_pkts = random.randint(0, 5)
            pkt_len_mean = random.uniform(40, 100)
            flow_pkts_sec = fwd_pkts / max(0.001, duration)
            syn_cnt = random.randint(5, 50)
        elif lbl == "PortScan":
            duration = random.uniform(0.001, 0.1)
            fwd_pkts = random.randint(1, 3)
            bwd_pkts = random.randint(0, 1)
            pkt_len_mean = random.uniform(40, 60)
            flow_pkts_sec = random.uniform(10, 100)
            syn_cnt = 1
        elif lbl == "Brute Force":
            duration = random.uniform(1.0, 10.0)
            fwd_pkts = random.randint(10, 100)
            bwd_pkts = random.randint(10, 80)
            pkt_len_mean = random.uniform(80, 200)
            flow_pkts_sec = random.uniform(5, 30)
            syn_cnt = 1
        else:
            duration = random.uniform(0.1, 5.0)
            fwd_pkts = random.randint(2, 30)
            bwd_pkts = random.randint(1, 30)
            pkt_len_mean = random.uniform(60, 500)
            flow_pkts_sec = random.uniform(1, 50)
            syn_cnt = 0

        tot_fwd_bytes = fwd_pkts * pkt_len_mean
        tot_bwd_bytes = bwd_pkts * (pkt_len_mean * random.uniform(0.8, 1.2))

        row = {
            "flow_duration": duration,
            "protocol_type": proto,
            "src_port": random.randint(1024, 65535),
            "dst_port": random.choice([80, 443, 22, 21, 53, 8080]),
            "fwd_packets": fwd_pkts,
            "bwd_packets": bwd_pkts,
            "total_fwd_bytes": tot_fwd_bytes,
            "total_bwd_bytes": tot_bwd_bytes,
            "fwd_pkt_len_max": pkt_len_mean * 1.5,
            "fwd_pkt_len_min": 40.0,
            "fwd_pkt_len_mean": pkt_len_mean,
            "fwd_pkt_len_std": pkt_len_mean * 0.2,
            "bwd_pkt_len_max": pkt_len_mean * 1.8,
            "bwd_pkt_len_min": 40.0,
            "bwd_pkt_len_mean": pkt_len_mean,
            "bwd_pkt_len_std": pkt_len_mean * 0.25,
            "flow_bytes_per_sec": (tot_fwd_bytes + tot_bwd_bytes) / max(0.001, duration),
            "flow_pkts_per_sec": flow_pkts_sec,
            "flow_iat_mean": duration / max(1, fwd_pkts + bwd_pkts),
            "flow_iat_std": 0.05,
            "flow_iat_max": duration * 0.8,
            "flow_iat_min": 0.0001,
            "fwd_iat_mean": duration / max(1, fwd_pkts),
            "fwd_iat_std": 0.04,
            "fwd_iat_max": duration * 0.7,
            "fwd_iat_min": 0.0001,
            "bwd_iat_mean": duration / max(1, bwd_pkts),
            "bwd_iat_std": 0.04,
            "bwd_iat_max": duration * 0.7,
            "bwd_iat_min": 0.0001,
            "fwd_psh_flags": 1 if is_attack else 0,
            "bwd_psh_flags": 0,
            "fwd_urg_flags": 0,
            "bwd_urg_flags": 0,
            "fin_flag_cnt": 1 if random.random() > 0.5 else 0,
            "syn_flag_cnt": syn_cnt,
            "rst_flag_cnt": 1 if is_attack and random.random() > 0.7 else 0,
            "psh_flag_cnt": 1 if random.random() > 0.4 else 0,
            "ack_flag_cnt": 1,
            "urg_flag_cnt": 0,
            "ece_flag_cnt": 0,
            "cwe_flag_cnt": 0,
            "pkt_len_mean": pkt_len_mean,
            "pkt_len_std": pkt_len_mean * 0.3,
            "pkt_len_var": (pkt_len_mean * 0.3) ** 2,
            "avg_pkt_size": pkt_len_mean,
            "fwd_seg_size_avg": pkt_len_mean,
            "bwd_seg_size_avg": pkt_len_mean,
            "subflow_fwd_pkts": fwd_pkts,
            "subflow_bwd_pkts": bwd_pkts,
            "subflow_fwd_bytes": tot_fwd_bytes,
            "subflow_bwd_bytes": tot_bwd_bytes,
            "init_win_bytes_fwd": random.choice([29200, 65535, 14600]),
            "init_win_bytes_bwd": random.choice([29200, 65535, 14600]),
            "act_data_pkt_fwd": fwd_pkts - 1,
            "min_seg_size_fwd": 20,
            "active_mean": duration * 0.8,
            "active_std": 0.0,
            "active_max": duration * 0.8,
            "active_min": duration * 0.8,
            "idle_mean": duration * 0.2,
            "idle_std": 0.0,
            "idle_max": duration * 0.2,
            "idle_min": duration * 0.2,
            "conn_frequency": random.randint(1, 20),
            "Label": lbl,
        }
        data.append(row)

    df = pd.DataFrame(data)
    df.to_csv(out_path, index=False)
    print(f"Generated {n_samples} synthetic network flow records at {out_path}")
    return out_path


def seed_database():
    """Initializes DB and inserts sample flows and incidents."""
    init_db()
    db = SessionLocal()

    if db.query(NetworkFlow).count() > 0:
        print("Database already has records. Skipping initial seeding.")
        db.close()
        return

    print("Seeding database with demonstration flows and incidents...")
    src_ips = ["192.168.1.101", "192.168.1.102", "10.0.0.45", "172.16.0.12", "198.51.100.22"]
    dst_ips = ["192.168.1.1", "192.168.1.254", "10.0.0.1", "172.16.0.1"]

    for i in range(60):
        src = random.choice(src_ips)
        dst = random.choice(dst_ips)
        is_attack = random.random() < 0.35
        label = random.choice(["DoS", "PortScan", "Brute Force"]) if is_attack else "BENIGN"
        proto = random.choice(["tcp", "udp", "icmp"])

        flow = NetworkFlow(
            flow_id=f"flow-{i+1:04d}",
            src_ip=src,
            dst_ip=dst,
            src_port=random.randint(1024, 65535),
            dst_port=random.choice([80, 443, 22, 53]),
            protocol=proto,
            duration=round(random.uniform(0.05, 4.5), 3),
            total_fwd_packets=random.randint(2, 80),
            total_bwd_packets=random.randint(1, 50),
            total_bytes=random.randint(120, 15000),
            label=label,
            attack_type=label.lower() if is_attack else "normal",
            source="demo_seed",
            features={"flow_duration": 1.2, "protocol_type": proto, "fwd_packets": 10, "bwd_packets": 5}
        )
        db.add(flow)
        db.flush()

        if is_attack:
            conf = round(random.uniform(0.78, 0.98), 2)
            anom = round(random.uniform(0.60, 0.95), 2)
            risk = round(conf * 40 + (85 if label == "DoS" else 55) * 0.30 + anom * 20 + 5, 1)
            sev = "critical" if risk >= 80 else ("high" if risk >= 60 else "medium")

            inc = DetectionIncident(
                flow_id=flow.id,
                detection_method=random.choice(["hybrid", "ml", "signature"]),
                predicted_label=label,
                attack_type=label,
                model_confidence=conf,
                anomaly_score=anom,
                risk_score=risk,
                severity=sev,
                src_ip=src,
                dst_ip=dst,
                src_port=flow.src_port,
                dst_port=flow.dst_port,
                protocol=proto,
                evidence={"model_confidence": conf, "anomaly_score": anom, "attack": label},
                shap_values={
                    "flow_pkts_per_sec": 0.34,
                    "syn_flag_cnt": 0.28,
                    "flow_duration": -0.15,
                    "fwd_pkt_len_mean": 0.12,
                    "total_fwd_bytes": 0.09
                },
                status="open"
            )
            db.add(inc)

    # Add a sample mitigation block
    block = MitigationAction(
        action_type="block",
        target_ip="198.51.100.22",
        reason="Repeated high-volume DoS flood attack",
        duration_minutes=60,
        is_active=True,
        is_simulated=True,
        status="active"
    )
    db.add(block)
    db.commit()
    db.close()
    print("Database seeded successfully with initial flows and incidents.")


if __name__ == "__main__":
    generate_synthetic_dataset(500, "sample_traffic.csv")
    seed_database()
