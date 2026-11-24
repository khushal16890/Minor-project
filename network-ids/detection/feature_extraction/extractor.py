"""
Feature extraction module for the Network Intrusion Detection System.

Extracts features from raw network data sources (Scapy packets, Zeek conn logs,
PCAP metadata) and returns dictionaries conforming to FEATURE_SCHEMA.
"""

import logging
import statistics
from typing import Any, Optional

from backend.core.config import FEATURE_SCHEMA

logger = logging.getLogger(__name__)

# Protocol number → readable name mapping (IP protocol field)
_PROTO_MAP = {
    1: "icmp",
    6: "tcp",
    17: "udp",
    47: "gre",
    50: "esp",
    58: "icmpv6",
}

# Zeek proto string → integer code
_ZEEK_PROTO_MAP = {
    "tcp": 6,
    "udp": 17,
    "icmp": 1,
}


def _safe_stat(values: list[float], func: str) -> float:
    """Return a statistical measure or 0.0 if the list is too short."""
    if not values:
        return 0.0
    if func == "mean":
        return statistics.mean(values)
    if func == "std":
        return statistics.pstdev(values) if len(values) > 1 else 0.0
    if func == "max":
        return max(values)
    if func == "min":
        return min(values)
    if func == "var":
        return statistics.pvariance(values) if len(values) > 1 else 0.0
    return 0.0


def _empty_feature_dict() -> dict[str, float]:
    """Return a feature dict with all FEATURE_SCHEMA keys set to 0.0."""
    return {f: 0.0 for f in FEATURE_SCHEMA}


class FeatureExtractor:
    """Extract network flow features from various raw data sources.

    Supported sources:
    * **Scapy packets** – a list of ``scapy.packet.Packet`` objects belonging
      to a single flow.
    * **Zeek conn logs** – a single parsed conn-log record (dict).
    * **PCAP metadata** – a summary dict produced by an external PCAP parser.

    Every method returns a ``dict[str, float]`` whose keys match
    ``FEATURE_SCHEMA`` from ``backend.core.config``.
    """

    # Columns that must never leak into model features
    _EXCLUDED_COLS = {"src_ip", "dst_ip", "timestamp", "flow_id"}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def extract_from_dict(self, raw: dict) -> dict[str, float]:
        """Align an incoming dictionary to FEATURE_SCHEMA, filling missing with 0.0."""
        features = _empty_feature_dict()
        for k, v in raw.items():
            if k in features:
                try:
                    features[k] = float(v)
                except (ValueError, TypeError):
                    features[k] = 0.0
        return features

    def extract_from_packets(
        self,
        packets: list,
        *,
        flow_id: Optional[str] = None,
    ) -> dict[str, float]:
        """Extract features from a list of Scapy packets belonging to one flow.

        Parameters
        ----------
        packets:
            Ordered list of ``scapy.packet.Packet`` objects for a single flow
            (same 5-tuple).  The first packet determines the forward direction.
        flow_id:
            Optional human-readable identifier (not used as a feature).

        Returns
        -------
        dict[str, float]
            Feature vector conforming to ``FEATURE_SCHEMA``.
        """
        features = _empty_feature_dict()

        if not packets:
            logger.warning("extract_from_packets called with empty packet list")
            return features

        try:
            first_pkt = packets[0]
            last_pkt = packets[-1]

            # Determine direction from first packet
            src_ip = self._get_ip_field(first_pkt, "src")
            src_port = self._get_port(first_pkt, "sport")
            dst_port = self._get_port(first_pkt, "dport")

            # Timestamps
            timestamps: list[float] = [float(p.time) for p in packets]
            flow_start = timestamps[0]
            flow_end = timestamps[-1]
            flow_duration = max(flow_end - flow_start, 0.0)
            features["flow_duration"] = flow_duration

            # Protocol
            features["protocol_type"] = float(self._get_protocol_num(first_pkt))
            features["src_port"] = float(src_port)
            features["dst_port"] = float(dst_port)

            # Separate fwd / bwd packets
            fwd_pkts: list = []
            bwd_pkts: list = []
            fwd_times: list[float] = []
            bwd_times: list[float] = []

            for i, pkt in enumerate(packets):
                pkt_src = self._get_ip_field(pkt, "src")
                if pkt_src == src_ip:
                    fwd_pkts.append(pkt)
                    fwd_times.append(timestamps[i])
                else:
                    bwd_pkts.append(pkt)
                    bwd_times.append(timestamps[i])

            features["fwd_packets"] = float(len(fwd_pkts))
            features["bwd_packets"] = float(len(bwd_pkts))

            # Byte counts
            fwd_bytes = [float(len(p)) for p in fwd_pkts]
            bwd_bytes = [float(len(p)) for p in bwd_pkts]
            features["total_fwd_bytes"] = sum(fwd_bytes)
            features["total_bwd_bytes"] = sum(bwd_bytes)

            # Packet length stats — forward
            features["fwd_pkt_len_max"] = _safe_stat(fwd_bytes, "max")
            features["fwd_pkt_len_min"] = _safe_stat(fwd_bytes, "min")
            features["fwd_pkt_len_mean"] = _safe_stat(fwd_bytes, "mean")
            features["fwd_pkt_len_std"] = _safe_stat(fwd_bytes, "std")

            # Packet length stats — backward
            features["bwd_pkt_len_max"] = _safe_stat(bwd_bytes, "max")
            features["bwd_pkt_len_min"] = _safe_stat(bwd_bytes, "min")
            features["bwd_pkt_len_mean"] = _safe_stat(bwd_bytes, "mean")
            features["bwd_pkt_len_std"] = _safe_stat(bwd_bytes, "std")

            # Overall packet length stats
            all_bytes = fwd_bytes + bwd_bytes
            features["pkt_len_mean"] = _safe_stat(all_bytes, "mean")
            features["pkt_len_std"] = _safe_stat(all_bytes, "std")
            features["pkt_len_var"] = _safe_stat(all_bytes, "var")
            total_pkts = len(all_bytes)
            total_bytes = sum(all_bytes)
            features["avg_pkt_size"] = total_bytes / total_pkts if total_pkts else 0.0

            # Flow rates
            if flow_duration > 0:
                features["flow_bytes_per_sec"] = total_bytes / flow_duration
                features["flow_pkts_per_sec"] = float(total_pkts) / flow_duration
            else:
                features["flow_bytes_per_sec"] = 0.0
                features["flow_pkts_per_sec"] = 0.0

            # Inter-arrival times
            self._compute_iat(timestamps, features, "flow")
            self._compute_iat(fwd_times, features, "fwd")
            self._compute_iat(bwd_times, features, "bwd")

            # TCP flags
            self._compute_tcp_flags(packets, fwd_pkts, bwd_pkts, features)

            # Segment size averages
            features["fwd_seg_size_avg"] = features["fwd_pkt_len_mean"]
            features["bwd_seg_size_avg"] = features["bwd_pkt_len_mean"]

            # Subflow (same as main flow for single-subflow extraction)
            features["subflow_fwd_pkts"] = features["fwd_packets"]
            features["subflow_bwd_pkts"] = features["bwd_packets"]
            features["subflow_fwd_bytes"] = features["total_fwd_bytes"]
            features["subflow_bwd_bytes"] = features["total_bwd_bytes"]

            # Init window bytes
            features["init_win_bytes_fwd"] = float(
                self._get_tcp_window(fwd_pkts[0]) if fwd_pkts else 0
            )
            features["init_win_bytes_bwd"] = float(
                self._get_tcp_window(bwd_pkts[0]) if bwd_pkts else 0
            )

            # Active data packets fwd (non-zero payload)
            features["act_data_pkt_fwd"] = float(
                sum(1 for p in fwd_pkts if self._payload_len(p) > 0)
            )

            # Min segment size forward
            if fwd_bytes:
                features["min_seg_size_fwd"] = min(fwd_bytes)

            # Active / idle times
            self._compute_active_idle(timestamps, features)

            # Connection frequency (default 1 for a single flow)
            features["conn_frequency"] = 1.0

        except Exception:
            logger.exception("Error extracting features from packets")

        return features

    def extract_from_zeek(self, conn_record: dict[str, Any]) -> dict[str, float]:
        """Extract features from a Zeek conn.log record.

        Parameters
        ----------
        conn_record:
            A dictionary parsed from a Zeek ``conn.log`` line (TSV or JSON).
            Expected keys include ``ts``, ``duration``, ``proto``, ``id.orig_p``,
            ``id.resp_p``, ``orig_bytes``, ``resp_bytes``, ``orig_pkts``,
            ``resp_pkts``, etc.

        Returns
        -------
        dict[str, float]
            Feature vector conforming to ``FEATURE_SCHEMA``.
        """
        features = _empty_feature_dict()

        try:
            duration = float(conn_record.get("duration", 0) or 0)
            features["flow_duration"] = duration

            proto_str = str(conn_record.get("proto", "tcp")).lower()
            features["protocol_type"] = float(_ZEEK_PROTO_MAP.get(proto_str, 0))

            features["src_port"] = float(conn_record.get("id.orig_p", 0) or 0)
            features["dst_port"] = float(conn_record.get("id.resp_p", 0) or 0)

            orig_pkts = float(conn_record.get("orig_pkts", 0) or 0)
            resp_pkts = float(conn_record.get("resp_pkts", 0) or 0)
            features["fwd_packets"] = orig_pkts
            features["bwd_packets"] = resp_pkts

            orig_bytes = float(conn_record.get("orig_bytes", 0) or 0)
            resp_bytes = float(conn_record.get("resp_bytes", 0) or 0)
            orig_ip_bytes = float(conn_record.get("orig_ip_bytes", 0) or 0)
            resp_ip_bytes = float(conn_record.get("resp_ip_bytes", 0) or 0)

            features["total_fwd_bytes"] = orig_bytes
            features["total_bwd_bytes"] = resp_bytes

            total_bytes = orig_bytes + resp_bytes
            total_pkts = orig_pkts + resp_pkts

            # Packet length estimates (Zeek gives totals, not per-packet)
            if orig_pkts > 0:
                avg_fwd = orig_ip_bytes / orig_pkts
                features["fwd_pkt_len_max"] = avg_fwd
                features["fwd_pkt_len_min"] = avg_fwd
                features["fwd_pkt_len_mean"] = avg_fwd
                features["fwd_seg_size_avg"] = avg_fwd
            if resp_pkts > 0:
                avg_bwd = resp_ip_bytes / resp_pkts
                features["bwd_pkt_len_max"] = avg_bwd
                features["bwd_pkt_len_min"] = avg_bwd
                features["bwd_pkt_len_mean"] = avg_bwd
                features["bwd_seg_size_avg"] = avg_bwd

            if total_pkts > 0:
                avg_all = (orig_ip_bytes + resp_ip_bytes) / total_pkts
                features["pkt_len_mean"] = avg_all
                features["avg_pkt_size"] = avg_all

            if duration > 0:
                features["flow_bytes_per_sec"] = total_bytes / duration
                features["flow_pkts_per_sec"] = total_pkts / duration
                # Rough IAT estimates
                if total_pkts > 1:
                    iat_est = duration / (total_pkts - 1)
                    for prefix in ("flow", "fwd", "bwd"):
                        features[f"{prefix}_iat_mean"] = iat_est
                        features[f"{prefix}_iat_max"] = iat_est
                        features[f"{prefix}_iat_min"] = iat_est

            # Subflow
            features["subflow_fwd_pkts"] = orig_pkts
            features["subflow_bwd_pkts"] = resp_pkts
            features["subflow_fwd_bytes"] = orig_bytes
            features["subflow_bwd_bytes"] = resp_bytes

            # TCP flag history parsing (Zeek uses chars: S, A, F, R, P, etc.)
            history = str(conn_record.get("history", ""))
            features["syn_flag_cnt"] = float(history.upper().count("S"))
            features["fin_flag_cnt"] = float(history.upper().count("F"))
            features["rst_flag_cnt"] = float(history.upper().count("R"))
            features["ack_flag_cnt"] = float(history.upper().count("A"))
            features["psh_flag_cnt"] = float(history.upper().count("P"))

            # Active / idle — no per-packet timestamps from Zeek, estimate
            features["active_mean"] = duration
            features["active_max"] = duration
            features["active_min"] = duration

            features["conn_frequency"] = 1.0

        except Exception:
            logger.exception("Error extracting features from Zeek conn record")

        return features

    def extract_from_pcap_metadata(self, metadata: dict[str, Any]) -> dict[str, float]:
        """Extract features from a pre-parsed PCAP metadata summary.

        Parameters
        ----------
        metadata:
            Dictionary with pre-computed flow statistics, typically produced by
            an external PCAP parsing utility.  Expected keys mirror the schema
            names (e.g. ``flow_duration``, ``protocol_type``, ``fwd_packets``).

        Returns
        -------
        dict[str, float]
            Feature vector conforming to ``FEATURE_SCHEMA``.
        """
        features = _empty_feature_dict()

        try:
            for key in FEATURE_SCHEMA:
                if key in metadata:
                    features[key] = float(metadata[key])

            # Derive missing aggregate stats if raw lists are provided
            if "packet_lengths" in metadata:
                pkt_lens = [float(v) for v in metadata["packet_lengths"]]
                features.setdefault("pkt_len_mean", _safe_stat(pkt_lens, "mean"))
                features["pkt_len_mean"] = features["pkt_len_mean"] or _safe_stat(pkt_lens, "mean")
                features["pkt_len_std"] = features["pkt_len_std"] or _safe_stat(pkt_lens, "std")
                features["pkt_len_var"] = features["pkt_len_var"] or _safe_stat(pkt_lens, "var")
                if pkt_lens:
                    features["avg_pkt_size"] = features["avg_pkt_size"] or (
                        sum(pkt_lens) / len(pkt_lens)
                    )

            if "inter_arrival_times" in metadata:
                iats = [float(v) for v in metadata["inter_arrival_times"]]
                for prefix in ("flow",):
                    features[f"{prefix}_iat_mean"] = features[f"{prefix}_iat_mean"] or _safe_stat(iats, "mean")
                    features[f"{prefix}_iat_std"] = features[f"{prefix}_iat_std"] or _safe_stat(iats, "std")
                    features[f"{prefix}_iat_max"] = features[f"{prefix}_iat_max"] or _safe_stat(iats, "max")
                    features[f"{prefix}_iat_min"] = features[f"{prefix}_iat_min"] or _safe_stat(iats, "min")

        except Exception:
            logger.exception("Error extracting features from PCAP metadata")

        return features

    def extract(self, raw_data: dict[str, Any]) -> dict[str, float]:
        """Auto-dispatch to the right extractor based on ``raw_data["source"]``.

        Parameters
        ----------
        raw_data:
            Must contain a ``"source"`` key with one of ``"packets"``,
            ``"zeek"``, or ``"pcap_metadata"``, plus the corresponding payload
            under ``"data"``.

        Returns
        -------
        dict[str, float]
        """
        source = raw_data.get("source", "").lower()
        data = raw_data.get("data")

        if source == "packets":
            return self.extract_from_packets(data)
        elif source == "zeek":
            return self.extract_from_zeek(data)
        elif source in ("pcap_metadata", "pcap"):
            return self.extract_from_pcap_metadata(data)
        else:
            logger.error("Unknown data source '%s'; returning empty features", source)
            return _empty_feature_dict()

    def validate_features(self, features: dict[str, float]) -> list[str]:
        """Check a feature dict for missing or unexpected keys.

        Returns a list of warning strings (empty if everything is valid).
        """
        warnings: list[str] = []
        for key in FEATURE_SCHEMA:
            if key not in features:
                warnings.append(f"Missing feature: {key}")
        extra = set(features.keys()) - set(FEATURE_SCHEMA)
        if extra:
            warnings.append(f"Extra (non-schema) features: {sorted(extra)}")
        return warnings

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _get_ip_field(pkt, field: str) -> str:
        """Get src or dst IP from a Scapy packet, handling IPv4/IPv6."""
        try:
            if pkt.haslayer("IP"):
                return getattr(pkt["IP"], field, "0.0.0.0")
            if pkt.haslayer("IPv6"):
                return getattr(pkt["IPv6"], field, "::")
        except Exception:
            pass
        return "0.0.0.0"

    @staticmethod
    def _get_port(pkt, field: str) -> int:
        """Get sport or dport from a Scapy packet."""
        try:
            if pkt.haslayer("TCP"):
                return int(getattr(pkt["TCP"], field, 0))
            if pkt.haslayer("UDP"):
                return int(getattr(pkt["UDP"], field, 0))
        except Exception:
            pass
        return 0

    @staticmethod
    def _get_protocol_num(pkt) -> int:
        """Return the IP protocol number from a Scapy packet."""
        try:
            if pkt.haslayer("IP"):
                return int(pkt["IP"].proto)
            if pkt.haslayer("IPv6"):
                return int(pkt["IPv6"].nh)
        except Exception:
            pass
        return 0

    @staticmethod
    def _get_tcp_window(pkt) -> int:
        """Return the TCP window size, or 0 if not TCP."""
        try:
            if pkt.haslayer("TCP"):
                return int(pkt["TCP"].window)
        except Exception:
            pass
        return 0

    @staticmethod
    def _payload_len(pkt) -> int:
        """Return payload (application-layer) length."""
        try:
            if pkt.haslayer("Raw"):
                return len(pkt["Raw"].load)
        except Exception:
            pass
        return 0

    @staticmethod
    def _compute_iat(
        timestamps: list[float],
        features: dict[str, float],
        prefix: str,
    ) -> None:
        """Compute inter-arrival time statistics and store in *features*."""
        if len(timestamps) < 2:
            return
        iats = [timestamps[i + 1] - timestamps[i] for i in range(len(timestamps) - 1)]
        features[f"{prefix}_iat_mean"] = _safe_stat(iats, "mean")
        features[f"{prefix}_iat_std"] = _safe_stat(iats, "std")
        features[f"{prefix}_iat_max"] = _safe_stat(iats, "max")
        features[f"{prefix}_iat_min"] = _safe_stat(iats, "min")

    @staticmethod
    def _compute_tcp_flags(
        all_pkts: list,
        fwd_pkts: list,
        bwd_pkts: list,
        features: dict[str, float],
    ) -> None:
        """Extract TCP flag counts from Scapy packets."""
        flag_map = {
            "F": "fin_flag_cnt",
            "S": "syn_flag_cnt",
            "R": "rst_flag_cnt",
            "P": "psh_flag_cnt",
            "A": "ack_flag_cnt",
            "U": "urg_flag_cnt",
            "E": "ece_flag_cnt",
            "C": "cwe_flag_cnt",
        }

        fwd_psh = 0
        bwd_psh = 0
        fwd_urg = 0
        bwd_urg = 0

        for pkt in all_pkts:
            try:
                if not pkt.haslayer("TCP"):
                    continue
                flags_str = str(pkt["TCP"].flags)
                for char, feat_name in flag_map.items():
                    if char in flags_str:
                        features[feat_name] += 1.0
            except Exception:
                continue

        for pkt in fwd_pkts:
            try:
                if not pkt.haslayer("TCP"):
                    continue
                flags_str = str(pkt["TCP"].flags)
                if "P" in flags_str:
                    fwd_psh += 1
                if "U" in flags_str:
                    fwd_urg += 1
            except Exception:
                continue

        for pkt in bwd_pkts:
            try:
                if not pkt.haslayer("TCP"):
                    continue
                flags_str = str(pkt["TCP"].flags)
                if "P" in flags_str:
                    bwd_psh += 1
                if "U" in flags_str:
                    bwd_urg += 1
            except Exception:
                continue

        features["fwd_psh_flags"] = float(fwd_psh)
        features["bwd_psh_flags"] = float(bwd_psh)
        features["fwd_urg_flags"] = float(fwd_urg)
        features["bwd_urg_flags"] = float(bwd_urg)

    @staticmethod
    def _compute_active_idle(
        timestamps: list[float],
        features: dict[str, float],
        idle_threshold: float = 1.0,
    ) -> None:
        """Estimate active/idle periods from packet timestamps.

        An *idle* period is any inter-packet gap larger than ``idle_threshold``
        seconds.  Everything else is considered *active*.
        """
        if len(timestamps) < 2:
            return

        active_periods: list[float] = []
        idle_periods: list[float] = []
        current_active = 0.0

        for i in range(len(timestamps) - 1):
            gap = timestamps[i + 1] - timestamps[i]
            if gap > idle_threshold:
                if current_active > 0:
                    active_periods.append(current_active)
                idle_periods.append(gap)
                current_active = 0.0
            else:
                current_active += gap

        if current_active > 0:
            active_periods.append(current_active)

        features["active_mean"] = _safe_stat(active_periods, "mean")
        features["active_std"] = _safe_stat(active_periods, "std")
        features["active_max"] = _safe_stat(active_periods, "max")
        features["active_min"] = _safe_stat(active_periods, "min")

        features["idle_mean"] = _safe_stat(idle_periods, "mean")
        features["idle_std"] = _safe_stat(idle_periods, "std")
        features["idle_max"] = _safe_stat(idle_periods, "max")
        features["idle_min"] = _safe_stat(idle_periods, "min")
