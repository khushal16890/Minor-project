"""
Suricata EVE JSON log parser.

Reads Suricata's ``eve.json`` (line-delimited JSON), extracts alert
records with signature metadata and flow 5-tuple, and optionally
creates ``SuricataAlert`` database records.

Supports:
- One-shot file parsing
- File tailing for live / near-real-time monitoring
"""

import os
import json
import time
import logging
import datetime
import threading
from typing import Optional, Callable

from backend.database.models import SuricataAlert, SessionLocal

logger = logging.getLogger(__name__)

# ─────────────────────── helpers ───────────────────────

_PROTO_MAP = {
    "TCP": "TCP", "UDP": "UDP", "ICMP": "ICMP",
    "6": "TCP", "17": "UDP", "1": "ICMP",
}


def _parse_timestamp(raw: str) -> Optional[datetime.datetime]:
    """Parse Suricata ISO 8601 timestamp (with or without TZ offset)."""
    for fmt in (
        "%Y-%m-%dT%H:%M:%S.%f%z",
        "%Y-%m-%dT%H:%M:%S%z",
        "%Y-%m-%dT%H:%M:%S.%f",
        "%Y-%m-%dT%H:%M:%S",
    ):
        try:
            return datetime.datetime.strptime(raw, fmt)
        except (ValueError, TypeError):
            continue
    return None


def _normalise_proto(raw) -> str:
    """Convert protocol to canonical upper-case name."""
    if raw is None:
        return ""
    return _PROTO_MAP.get(str(raw).upper(), str(raw).upper())


# ─────────────────────── data class ───────────────────────


class ParsedAlert:
    """Lightweight container for a parsed Suricata alert."""
    __slots__ = (
        "timestamp", "signature_id", "signature", "category",
        "severity", "src_ip", "dst_ip", "src_port", "dst_port",
        "protocol", "action", "flow_id", "payload", "raw_event",
    )

    def __init__(
        self,
        timestamp: Optional[datetime.datetime] = None,
        signature_id: int = 0,
        signature: str = "",
        category: str = "",
        severity: int = 0,
        src_ip: str = "",
        dst_ip: str = "",
        src_port: int = 0,
        dst_port: int = 0,
        protocol: str = "",
        action: str = "",
        flow_id: str = "",
        payload: str = "",
        raw_event: Optional[dict] = None,
    ):
        self.timestamp = timestamp
        self.signature_id = signature_id
        self.signature = signature
        self.category = category
        self.severity = severity
        self.src_ip = src_ip
        self.dst_ip = dst_ip
        self.src_port = src_port
        self.dst_port = dst_port
        self.protocol = protocol
        self.action = action
        self.flow_id = flow_id
        self.payload = payload
        self.raw_event = raw_event

    def to_dict(self) -> dict:
        return {
            "timestamp": self.timestamp.isoformat() if self.timestamp else None,
            "signature_id": self.signature_id,
            "signature": self.signature,
            "category": self.category,
            "severity": self.severity,
            "src_ip": self.src_ip,
            "dst_ip": self.dst_ip,
            "src_port": self.src_port,
            "dst_port": self.dst_port,
            "protocol": self.protocol,
            "action": self.action,
            "flow_id": self.flow_id,
        }


# ─────────────────────── SuricataParser ───────────────────────


class SuricataParser:
    """Parse Suricata EVE JSON logs and extract alerts.

    Parameters
    ----------
    eve_path : str, optional
        Default path to ``eve.json``.  Can be overridden per method call.

    Examples
    --------
    >>> parser = SuricataParser("/var/log/suricata/eve.json")
    >>> alerts = parser.parse_file()
    >>> parser.save_alerts_to_db(alerts)
    """

    def __init__(self, eve_path: Optional[str] = None):
        self.eve_path = eve_path
        self._tail_running = False
        self._tail_thread: Optional[threading.Thread] = None

    # ───── public API ─────

    def parse_file(self, path: Optional[str] = None) -> list[ParsedAlert]:
        """Parse an entire EVE JSON file and return alert records.

        Parameters
        ----------
        path : str, optional
            File to read.  Defaults to ``self.eve_path``.

        Returns
        -------
        list[ParsedAlert]
        """
        fpath = path or self.eve_path
        if not fpath or not os.path.isfile(fpath):
            raise FileNotFoundError(f"EVE JSON not found: {fpath}")

        alerts: list[ParsedAlert] = []
        line_count = 0

        with open(fpath, "r", encoding="utf-8", errors="replace") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                line_count += 1
                try:
                    event = json.loads(line)
                except json.JSONDecodeError:
                    logger.debug("Skipping invalid JSON line %d", line_count)
                    continue

                if event.get("event_type") == "alert":
                    alert = self._parse_alert_event(event)
                    if alert:
                        alerts.append(alert)

        logger.info("Parsed %d alerts from %d lines in %s",
                     len(alerts), line_count, fpath)
        return alerts

    def parse_all_events(self, path: Optional[str] = None) -> list[dict]:
        """Return *all* EVE events (not just alerts) as raw dicts."""
        fpath = path or self.eve_path
        if not fpath or not os.path.isfile(fpath):
            raise FileNotFoundError(f"EVE JSON not found: {fpath}")

        events = []
        with open(fpath, "r", encoding="utf-8", errors="replace") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    events.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
        return events

    def save_alerts_to_db(self, alerts: list[ParsedAlert]) -> int:
        """Persist a list of ``ParsedAlert`` objects to the database.

        Returns the number of rows committed.
        """
        session = SessionLocal()
        saved = 0
        try:
            for alert in alerts:
                record = SuricataAlert(
                    timestamp=alert.timestamp,
                    signature_id=alert.signature_id,
                    signature=alert.signature,
                    category=alert.category,
                    severity=alert.severity,
                    src_ip=alert.src_ip,
                    dst_ip=alert.dst_ip,
                    src_port=alert.src_port,
                    dst_port=alert.dst_port,
                    protocol=alert.protocol,
                    action=alert.action,
                    flow_id=alert.flow_id,
                    payload=alert.payload,
                    raw_event=alert.raw_event,
                )
                session.add(record)
                saved += 1
            session.commit()
            logger.info("Saved %d alerts to database", saved)
        except Exception:
            session.rollback()
            logger.exception("Failed to save alerts to database")
            raise
        finally:
            session.close()
        return saved

    # ───── live tailing ─────

    def start_tailing(
        self,
        path: Optional[str] = None,
        callback: Optional[Callable[[ParsedAlert], None]] = None,
        poll_interval: float = 1.0,
        save_to_db: bool = False,
    ) -> None:
        """Begin tailing an EVE JSON file for new alert lines.

        Runs on a background daemon thread.  Call :meth:`stop_tailing`
        to terminate.

        Parameters
        ----------
        path : str, optional
            File to tail.
        callback : callable, optional
            Called with each new ``ParsedAlert``.
        poll_interval : float
            Seconds between tail polls.
        save_to_db : bool
            Automatically persist new alerts to the database.
        """
        if self._tail_running:
            logger.warning("Tailing is already active")
            return

        fpath = path or self.eve_path
        if not fpath:
            raise ValueError("No EVE JSON path specified")

        self._tail_running = True
        self._tail_thread = threading.Thread(
            target=self._tail_loop,
            args=(fpath, callback, poll_interval, save_to_db),
            daemon=True,
        )
        self._tail_thread.start()
        logger.info("Started tailing %s", fpath)

    def stop_tailing(self) -> None:
        """Stop the background tail thread."""
        self._tail_running = False
        if self._tail_thread and self._tail_thread.is_alive():
            self._tail_thread.join(timeout=5)
        logger.info("Tailing stopped")

    # ───── internals ─────

    def _tail_loop(
        self,
        path: str,
        callback: Optional[Callable],
        poll_interval: float,
        save_to_db: bool,
    ) -> None:
        """Read new lines appended to the file."""
        offset = 0
        if os.path.isfile(path):
            offset = os.path.getsize(path)  # start from end

        while self._tail_running:
            try:
                if not os.path.isfile(path):
                    time.sleep(poll_interval)
                    continue

                current_size = os.path.getsize(path)
                if current_size < offset:
                    # File was truncated / rotated
                    offset = 0

                if current_size == offset:
                    time.sleep(poll_interval)
                    continue

                with open(path, "r", encoding="utf-8", errors="replace") as fh:
                    fh.seek(offset)
                    new_lines = fh.readlines()
                    offset = fh.tell()

                batch: list[ParsedAlert] = []
                for line in new_lines:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        event = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    if event.get("event_type") == "alert":
                        alert = self._parse_alert_event(event)
                        if alert:
                            batch.append(alert)
                            if callback:
                                try:
                                    callback(alert)
                                except Exception as exc:
                                    logger.debug("Tail callback error: %s", exc)

                if save_to_db and batch:
                    try:
                        self.save_alerts_to_db(batch)
                    except Exception as exc:
                        logger.error("DB save during tail failed: %s", exc)

            except Exception as exc:
                logger.error("Tail loop error: %s", exc)

            time.sleep(poll_interval)

    @staticmethod
    def _parse_alert_event(event: dict) -> Optional[ParsedAlert]:
        """Extract a ``ParsedAlert`` from a raw EVE JSON alert dict."""
        alert_data = event.get("alert", {})
        if not alert_data:
            return None

        return ParsedAlert(
            timestamp=_parse_timestamp(event.get("timestamp", "")),
            signature_id=alert_data.get("signature_id", 0),
            signature=alert_data.get("signature", ""),
            category=alert_data.get("category", ""),
            severity=alert_data.get("severity", 0),
            src_ip=event.get("src_ip", ""),
            dst_ip=event.get("dest_ip", event.get("dst_ip", "")),
            src_port=event.get("src_port", 0),
            dst_port=event.get("dest_port", event.get("dst_port", 0)),
            protocol=_normalise_proto(event.get("proto")),
            action=alert_data.get("action", ""),
            flow_id=str(event.get("flow_id", "")),
            payload=event.get("payload", ""),
            raw_event=event,
        )
