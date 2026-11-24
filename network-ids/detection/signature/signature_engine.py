"""
Signature-based detection engine.

Wraps Suricata alert ingestion, matches alerts to tracked network flows
where possible, and creates ``DetectionIncident`` records.  Provides
statistics on signature match rates.
"""

import logging
import datetime
from typing import Optional

from backend.core.config import get_attack_severity
from backend.database.models import (
    SuricataAlert,
    DetectionIncident,
    NetworkFlow,
    SessionLocal,
)
from network.suricata.suricata_parser import SuricataParser, ParsedAlert

logger = logging.getLogger(__name__)


# ─────────────────────── severity mapping ───────────────────────

def _suricata_severity_to_label(severity: int) -> str:
    """Map Suricata's 1-4 severity scale to internal label."""
    return {1: "critical", 2: "high", 3: "medium"}.get(severity, "low")


def _risk_score_from_severity(severity: int, confidence: float = 0.9) -> float:
    """Compute a 0-100 risk score from Suricata severity + confidence."""
    base = {1: 90, 2: 70, 3: 50}.get(severity, 30)
    return min(round(base * confidence, 2), 100.0)


# ─────────────────────── SignatureEngine ───────────────────────


class SignatureEngine:
    """Signature-based detection backed by Suricata alerts.

    Workflow
    --------
    1. Ingest alerts from an EVE JSON file (or live tail).
    2. For each alert, attempt to find a matching ``NetworkFlow`` in the
       database by IP/port/protocol/timestamp proximity.
    3. Create a ``DetectionIncident`` with ``detection_method='signature'``.
    4. Track match statistics.

    Parameters
    ----------
    parser : SuricataParser, optional
        Pre-configured parser instance.  One is created if not supplied.
    eve_path : str, optional
        Default path to Suricata ``eve.json``.
    time_window_seconds : float
        Max time delta (seconds) for matching an alert to a flow.
    """

    def __init__(
        self,
        parser: Optional[SuricataParser] = None,
        eve_path: Optional[str] = None,
        time_window_seconds: float = 60.0,
    ):
        self.parser = parser or SuricataParser(eve_path=eve_path)
        self.time_window = time_window_seconds

        # Statistics
        self._stats = {
            "total_alerts_processed": 0,
            "alerts_matched_to_flows": 0,
            "alerts_unmatched": 0,
            "incidents_created": 0,
            "by_category": {},
            "by_severity": {"critical": 0, "high": 0, "medium": 0, "low": 0},
        }

    # ───── public API ─────

    def ingest_from_file(
        self,
        path: Optional[str] = None,
        save_alerts: bool = True,
    ) -> list[dict]:
        """Parse an EVE JSON file and create detection incidents.

        Parameters
        ----------
        path : str, optional
            Path to ``eve.json``.
        save_alerts : bool
            Also persist the raw ``SuricataAlert`` rows.

        Returns
        -------
        list[dict]
            Created incident summaries.
        """
        alerts = self.parser.parse_file(path)
        if save_alerts:
            self.parser.save_alerts_to_db(alerts)

        return self.process_alerts(alerts)

    def process_alerts(self, alerts: list[ParsedAlert]) -> list[dict]:
        """Match a batch of ``ParsedAlert`` objects to flows and create incidents.

        Returns
        -------
        list[dict]
            One dict per created ``DetectionIncident``.
        """
        session = SessionLocal()
        incidents: list[dict] = []

        try:
            for alert in alerts:
                self._stats["total_alerts_processed"] += 1

                # Category stats
                cat = alert.category or "uncategorized"
                self._stats["by_category"][cat] = (
                    self._stats["by_category"].get(cat, 0) + 1
                )

                sev_label = _suricata_severity_to_label(alert.severity)
                self._stats["by_severity"][sev_label] += 1

                # Try to match to a network flow
                flow = self._find_matching_flow(session, alert)
                if flow:
                    self._stats["alerts_matched_to_flows"] += 1
                else:
                    self._stats["alerts_unmatched"] += 1

                # Create incident
                incident = self._create_incident(session, alert, flow)
                incidents.append(incident)
                self._stats["incidents_created"] += 1

            session.commit()
            logger.info(
                "Processed %d alerts → %d incidents (%d matched to flows)",
                len(alerts), len(incidents),
                self._stats["alerts_matched_to_flows"],
            )
        except Exception:
            session.rollback()
            logger.exception("Error processing signature alerts")
            raise
        finally:
            session.close()

        return incidents

    def start_live_monitoring(
        self,
        path: Optional[str] = None,
        poll_interval: float = 2.0,
    ) -> None:
        """Begin tailing the EVE JSON file, creating incidents in real time."""
        def _on_alert(alert: ParsedAlert):
            try:
                self.process_alerts([alert])
            except Exception as exc:
                logger.error("Live monitoring error: %s", exc)

        self.parser.start_tailing(
            path=path,
            callback=_on_alert,
            poll_interval=poll_interval,
            save_to_db=True,
        )

    def stop_live_monitoring(self) -> None:
        """Stop the live tail."""
        self.parser.stop_tailing()

    def get_stats(self) -> dict:
        """Return cumulative signature match statistics."""
        return dict(self._stats)

    def reset_stats(self) -> None:
        """Reset all counters."""
        self._stats = {
            "total_alerts_processed": 0,
            "alerts_matched_to_flows": 0,
            "alerts_unmatched": 0,
            "incidents_created": 0,
            "by_category": {},
            "by_severity": {"critical": 0, "high": 0, "medium": 0, "low": 0},
        }

    # ───── internals ─────

    def _find_matching_flow(
        self,
        session,
        alert: ParsedAlert,
    ) -> Optional[NetworkFlow]:
        """Find the best-matching ``NetworkFlow`` by 5-tuple + time window."""
        if not alert.src_ip or not alert.dst_ip:
            return None

        query = session.query(NetworkFlow).filter(
            NetworkFlow.src_ip == alert.src_ip,
            NetworkFlow.dst_ip == alert.dst_ip,
        )

        if alert.src_port:
            query = query.filter(NetworkFlow.src_port == alert.src_port)
        if alert.dst_port:
            query = query.filter(NetworkFlow.dst_port == alert.dst_port)
        if alert.protocol:
            query = query.filter(NetworkFlow.protocol == alert.protocol)

        if alert.timestamp:
            window_start = alert.timestamp - datetime.timedelta(
                seconds=self.time_window
            )
            window_end = alert.timestamp + datetime.timedelta(
                seconds=self.time_window
            )
            query = query.filter(
                NetworkFlow.timestamp >= window_start,
                NetworkFlow.timestamp <= window_end,
            )

        # Order by recency and pick the closest one
        query = query.order_by(NetworkFlow.timestamp.desc())
        return query.first()

    @staticmethod
    def _create_incident(
        session,
        alert: ParsedAlert,
        flow: Optional[NetworkFlow],
    ) -> dict:
        """Build and persist a ``DetectionIncident`` from a signature alert."""
        sev_label = _suricata_severity_to_label(alert.severity)
        risk = _risk_score_from_severity(alert.severity)

        # Guess attack type from category
        attack_type = alert.category or "unknown"

        incident = DetectionIncident(
            flow_id=flow.id if flow else None,
            timestamp=alert.timestamp or datetime.datetime.now(datetime.timezone.utc),
            detection_method="signature",
            predicted_label="malicious",
            attack_type=attack_type,
            model_confidence=0.9,  # signature = high confidence
            risk_score=risk,
            severity=sev_label,
            signature_id=str(alert.signature_id) if alert.signature_id else None,
            signature_msg=alert.signature,
            src_ip=alert.src_ip,
            dst_ip=alert.dst_ip,
            src_port=alert.src_port,
            dst_port=alert.dst_port,
            protocol=alert.protocol,
            evidence={
                "signature_id": alert.signature_id,
                "signature": alert.signature,
                "category": alert.category,
                "suricata_severity": alert.severity,
                "action": alert.action,
                "flow_id": alert.flow_id,
            },
            status="open",
        )
        session.add(incident)

        return {
            "signature_id": alert.signature_id,
            "signature": alert.signature,
            "category": alert.category,
            "severity": sev_label,
            "risk_score": risk,
            "src_ip": alert.src_ip,
            "dst_ip": alert.dst_ip,
            "matched_flow": flow.id if flow else None,
        }
