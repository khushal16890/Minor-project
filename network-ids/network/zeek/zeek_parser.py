"""
Zeek log parser for conn.log, dns.log, and http.log.

Handles the standard Zeek TSV format including comment lines (``#``),
``#separator``, ``#fields``, and ``#types`` directives.  Maps Zeek
connection metadata to the internal ``FEATURE_SCHEMA`` feature dict.
"""

import os
import csv
import logging
import datetime
from typing import Optional

from backend.core.config import FEATURE_SCHEMA

logger = logging.getLogger(__name__)

# ─────────────────────── Zeek field helpers ───────────────────────

# Zeek conn.log field → internal feature mapping
_CONN_FIELD_MAP = {
    "ts":          "_timestamp",
    "uid":         "_uid",
    "id.orig_h":   "_src_ip",
    "id.resp_h":   "_dst_ip",
    "id.orig_p":   "src_port",
    "id.resp_p":   "dst_port",
    "proto":       "_proto_str",
    "service":     "_service",
    "duration":    "flow_duration",
    "orig_bytes":  "total_fwd_bytes",
    "resp_bytes":  "total_bwd_bytes",
    "conn_state":  "_conn_state",
    "orig_pkts":   "fwd_packets",
    "resp_pkts":   "bwd_packets",
    "orig_ip_bytes": "_orig_ip_bytes",
    "resp_ip_bytes": "_resp_ip_bytes",
    "missed_bytes":  "_missed_bytes",
    "history":     "_history",
}

_PROTO_MAP = {"tcp": 6, "udp": 17, "icmp": 1}


def _safe_int(val: str, default: int = 0) -> int:
    if val in ("-", "(empty)", ""):
        return default
    try:
        return int(val)
    except (ValueError, TypeError):
        return default


def _safe_float(val: str, default: float = 0.0) -> float:
    if val in ("-", "(empty)", ""):
        return default
    try:
        return float(val)
    except (ValueError, TypeError):
        return default


# ─────────────────────── ZeekParser ───────────────────────


class ZeekParser:
    """Parse Zeek log files and convert records to internal feature dicts.

    Supports:
    - ``conn.log`` — connection records
    - ``dns.log``  — DNS query/response records
    - ``http.log`` — HTTP request records

    Examples
    --------
    >>> parser = ZeekParser()
    >>> flows = parser.parse_conn_log("/path/to/conn.log")
    >>> dns   = parser.parse_dns_log("/path/to/dns.log")
    >>> http  = parser.parse_http_log("/path/to/http.log")
    """

    def __init__(self):
        self._conn_records: list[dict] = []
        self._dns_records: list[dict] = []
        self._http_records: list[dict] = []

    # ───── public API ─────

    def parse_conn_log(self, path: str) -> list[dict]:
        """Parse a Zeek ``conn.log`` and return feature dicts.

        Each dict is compatible with ``FEATURE_SCHEMA`` (numerical
        features filled where data is available; the rest default to 0).

        Parameters
        ----------
        path : str
            Path to the Zeek ``conn.log`` file.

        Returns
        -------
        list[dict]
            One dict per connection record.
        """
        raw_records = self._read_zeek_log(path)
        logger.info("Parsed %d records from conn.log: %s", len(raw_records), path)

        features_list = []
        for rec in raw_records:
            features_list.append(self._conn_record_to_features(rec))

        self._conn_records = features_list
        return features_list

    def parse_dns_log(self, path: str) -> list[dict]:
        """Parse a Zeek ``dns.log``.

        Returns
        -------
        list[dict]
            Dicts with keys: ``_timestamp``, ``_uid``, ``_src_ip``,
            ``_dst_ip``, ``src_port``, ``dst_port``, ``query``,
            ``qtype_name``, ``rcode_name``, ``answers``, ``TTLs``.
        """
        raw = self._read_zeek_log(path)
        logger.info("Parsed %d DNS records from %s", len(raw), path)

        results = []
        for rec in raw:
            entry = {
                "_timestamp": rec.get("ts", ""),
                "_uid": rec.get("uid", ""),
                "_src_ip": rec.get("id.orig_h", ""),
                "_dst_ip": rec.get("id.resp_h", ""),
                "src_port": _safe_int(rec.get("id.orig_p", "0")),
                "dst_port": _safe_int(rec.get("id.resp_p", "0")),
                "query": rec.get("query", ""),
                "qtype_name": rec.get("qtype_name", ""),
                "qtype": _safe_int(rec.get("qtype", "0")),
                "rcode": _safe_int(rec.get("rcode", "0")),
                "rcode_name": rec.get("rcode_name", ""),
                "answers": rec.get("answers", ""),
                "TTLs": rec.get("TTLs", ""),
                "rejected": rec.get("rejected", "F") == "T",
                "trans_id": _safe_int(rec.get("trans_id", "0")),
            }
            results.append(entry)

        self._dns_records = results
        return results

    def parse_http_log(self, path: str) -> list[dict]:
        """Parse a Zeek ``http.log``.

        Returns
        -------
        list[dict]
            Dicts with HTTP request metadata.
        """
        raw = self._read_zeek_log(path)
        logger.info("Parsed %d HTTP records from %s", len(raw), path)

        results = []
        for rec in raw:
            entry = {
                "_timestamp": rec.get("ts", ""),
                "_uid": rec.get("uid", ""),
                "_src_ip": rec.get("id.orig_h", ""),
                "_dst_ip": rec.get("id.resp_h", ""),
                "src_port": _safe_int(rec.get("id.orig_p", "0")),
                "dst_port": _safe_int(rec.get("id.resp_p", "0")),
                "method": rec.get("method", ""),
                "host": rec.get("host", ""),
                "uri": rec.get("uri", ""),
                "referrer": rec.get("referrer", ""),
                "user_agent": rec.get("user_agent", ""),
                "status_code": _safe_int(rec.get("status_code", "0")),
                "status_msg": rec.get("status_msg", ""),
                "request_body_len": _safe_int(rec.get("request_body_len", "0")),
                "response_body_len": _safe_int(rec.get("response_body_len", "0")),
                "content_type": rec.get("resp_mime_types", ""),
                "trans_depth": _safe_int(rec.get("trans_depth", "0")),
            }
            results.append(entry)

        self._http_records = results
        return results

    @property
    def conn_records(self) -> list[dict]:
        """Return the most recently parsed conn.log records."""
        return self._conn_records

    @property
    def dns_records(self) -> list[dict]:
        return self._dns_records

    @property
    def http_records(self) -> list[dict]:
        return self._http_records

    # ───── internal helpers ─────

    @staticmethod
    def _read_zeek_log(path: str) -> list[dict]:
        """Read a Zeek TSV log file respecting ``#`` comment directives.

        Automatically extracts ``#fields`` and ``#separator`` and returns
        a list of dicts keyed by field name.
        """
        if not os.path.isfile(path):
            raise FileNotFoundError(f"Zeek log not found: {path}")

        separator = "\t"
        fields: list[str] = []
        records: list[dict] = []

        with open(path, "r", encoding="utf-8", errors="replace") as fh:
            for line in fh:
                line = line.rstrip("\n\r")

                # Directive lines
                if line.startswith("#"):
                    if line.startswith("#separator"):
                        # e.g. #separator \x09
                        raw_sep = line.split(None, 1)[1] if " " in line else "\t"
                        separator = raw_sep.encode().decode("unicode_escape")
                    elif line.startswith("#fields"):
                        fields = line.split(separator)[1:]
                        # Strip leading/trailing whitespace from field names
                        fields = [f.strip() for f in fields]
                    # Skip #types, #open, #close, #empty_field, #unset_field, etc.
                    continue

                if not fields:
                    continue  # no header yet

                parts = line.split(separator)
                if len(parts) != len(fields):
                    # Tolerate minor mismatches at end of file
                    if len(parts) < len(fields):
                        parts.extend([""] * (len(fields) - len(parts)))
                    else:
                        parts = parts[:len(fields)]

                records.append(dict(zip(fields, parts)))

        return records

    @staticmethod
    def _conn_record_to_features(rec: dict) -> dict:
        """Convert one Zeek conn.log record dict into a ``FEATURE_SCHEMA``
        compatible feature dict.

        Fields that cannot be derived from the conn.log are set to 0.
        """
        duration = _safe_float(rec.get("duration", "0"))
        fwd_bytes = _safe_int(rec.get("orig_bytes", "0"))
        bwd_bytes = _safe_int(rec.get("resp_bytes", "0"))
        fwd_pkts = _safe_int(rec.get("orig_pkts", "0"))
        bwd_pkts = _safe_int(rec.get("resp_pkts", "0"))
        total_pkts = fwd_pkts + bwd_pkts
        total_bytes = fwd_bytes + bwd_bytes
        duration_safe = max(duration, 1e-6)

        proto_str = rec.get("proto", "").lower()
        proto_val = _PROTO_MAP.get(proto_str, 0)

        # Derive averages where possible
        fwd_mean = fwd_bytes / fwd_pkts if fwd_pkts else 0.0
        bwd_mean = bwd_bytes / bwd_pkts if bwd_pkts else 0.0
        avg_pkt = total_bytes / total_pkts if total_pkts else 0.0

        # Count flags from Zeek history string
        hist = rec.get("history", "")
        syn_cnt = hist.count("S") + hist.count("s")
        fin_cnt = hist.count("F") + hist.count("f")
        rst_cnt = hist.count("R") + hist.count("r")
        ack_cnt = hist.count("A") + hist.count("a")

        features: dict = {feat: 0 for feat in FEATURE_SCHEMA}
        features.update({
            "flow_duration": duration,
            "protocol_type": proto_val,
            "src_port": _safe_int(rec.get("id.orig_p", "0")),
            "dst_port": _safe_int(rec.get("id.resp_p", "0")),
            "fwd_packets": fwd_pkts,
            "bwd_packets": bwd_pkts,
            "total_fwd_bytes": fwd_bytes,
            "total_bwd_bytes": bwd_bytes,
            "fwd_pkt_len_mean": fwd_mean,
            "bwd_pkt_len_mean": bwd_mean,
            "flow_bytes_per_sec": total_bytes / duration_safe,
            "flow_pkts_per_sec": total_pkts / duration_safe,
            "pkt_len_mean": avg_pkt,
            "avg_pkt_size": avg_pkt,
            "fwd_seg_size_avg": fwd_mean,
            "bwd_seg_size_avg": bwd_mean,
            "subflow_fwd_pkts": fwd_pkts,
            "subflow_bwd_pkts": bwd_pkts,
            "subflow_fwd_bytes": fwd_bytes,
            "subflow_bwd_bytes": bwd_bytes,
            "syn_flag_cnt": syn_cnt,
            "fin_flag_cnt": fin_cnt,
            "rst_flag_cnt": rst_cnt,
            "ack_flag_cnt": ack_cnt,
            "active_mean": duration,
            "active_max": duration,
            "active_min": duration,
            "conn_frequency": 1.0,
        })

        # Metadata
        ts_raw = rec.get("ts", "")
        try:
            ts_float = float(ts_raw)
            ts_iso = datetime.datetime.fromtimestamp(
                ts_float, tz=datetime.timezone.utc
            ).isoformat()
        except (ValueError, TypeError):
            ts_iso = ts_raw

        features["_src_ip"] = rec.get("id.orig_h", "")
        features["_dst_ip"] = rec.get("id.resp_h", "")
        features["_timestamp"] = ts_iso
        features["_uid"] = rec.get("uid", "")
        features["_service"] = rec.get("service", "")
        features["_conn_state"] = rec.get("conn_state", "")
        features["_proto_str"] = proto_str

        return features
