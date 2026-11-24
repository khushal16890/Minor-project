"""
PCAP / PCAPNG file processor.

Reads packet capture files via Scapy, groups packets into flows by their
5-tuple (src_ip, dst_ip, src_port, dst_port, protocol), computes per-flow
statistics (duration, bytes, packets, IAT, TCP flags), and returns a list
of feature dicts compatible with ``FEATURE_SCHEMA``.

Large files are handled by batched reading to control memory usage.
"""

import os
import logging
import statistics
import datetime
from typing import Optional
from collections import defaultdict
from dataclasses import dataclass, field

try:
    from scapy.all import rdpcap, PcapReader, IP, IPv6, TCP, UDP, ICMP, DNS
except ImportError:
    raise ImportError(
        "Scapy is required. Install it with: pip install scapy"
    )

from backend.core.config import FEATURE_SCHEMA

logger = logging.getLogger(__name__)

# ─────────────────────── helpers ───────────────────────


def _proto_str(num: int) -> str:
    return {1: "ICMP", 6: "TCP", 17: "UDP"}.get(num, str(num))


def _port(pkt, layer, attr: str) -> int:
    if pkt.haslayer(layer):
        return int(getattr(pkt[layer], attr, 0))
    return 0


# ─────────────────────── per-flow accumulator ───────────────────────


@dataclass
class _FlowAcc:
    """Accumulates packet-level data for a single flow."""
    src_ip: str
    dst_ip: str
    src_port: int
    dst_port: int
    protocol: str
    start_ts: float = 0.0
    end_ts: float = 0.0
    fwd_pkts: int = 0
    bwd_pkts: int = 0
    fwd_bytes: int = 0
    bwd_bytes: int = 0
    fwd_sizes: list = field(default_factory=list)
    bwd_sizes: list = field(default_factory=list)
    all_ts: list = field(default_factory=list)
    fwd_ts: list = field(default_factory=list)
    bwd_ts: list = field(default_factory=list)
    fin: int = 0
    syn: int = 0
    rst: int = 0
    psh: int = 0
    ack: int = 0
    urg: int = 0
    ece: int = 0
    cwe: int = 0
    init_win_fwd: int = -1
    init_win_bwd: int = -1
    has_dns: bool = False
    has_http: bool = False


# ─────────────────────── PcapProcessor ───────────────────────


class PcapProcessor:
    """Read PCAP/PCAPNG files and extract per-flow features.

    Parameters
    ----------
    batch_size : int
        Number of packets to read per batch when processing large files.
    """

    def __init__(self, batch_size: int = 50_000):
        self.batch_size = batch_size

    # ───── public API ─────

    def process_file(self, path: str) -> list[dict]:
        """Process a PCAP/PCAPNG file and return flow feature dicts.

        Parameters
        ----------
        path : str
            Path to a ``.pcap`` or ``.pcapng`` file.

        Returns
        -------
        list[dict]
            One dict per flow with keys from ``FEATURE_SCHEMA`` plus
            metadata keys prefixed with ``_``.
        """
        if not os.path.isfile(path):
            raise FileNotFoundError(f"PCAP file not found: {path}")

        file_size_mb = os.path.getsize(path) / (1024 * 1024)
        logger.info("Processing PCAP %s (%.1f MB)", path, file_size_mb)

        flows = self._extract_flows(path)
        logger.info("Extracted %d flows from %s", len(flows), path)
        return [self._flow_to_features(acc) for acc in flows.values()]

    def get_summary(self, path: str) -> dict:
        """Return a quick summary without full feature extraction.

        Returns dict with ``total_packets``, ``flows``, ``protocols``,
        ``duration_seconds``, ``src_ips``, ``dst_ips``.
        """
        flows = self._extract_flows(path)
        total_pkts = sum(f.fwd_pkts + f.bwd_pkts for f in flows.values())
        protocols = set(f.protocol for f in flows.values())
        src_ips = set(f.src_ip for f in flows.values())
        dst_ips = set(f.dst_ip for f in flows.values())

        all_start = min((f.start_ts for f in flows.values()), default=0)
        all_end = max((f.end_ts for f in flows.values()), default=0)

        return {
            "total_packets": total_pkts,
            "flows": len(flows),
            "protocols": list(protocols),
            "duration_seconds": round(all_end - all_start, 3),
            "unique_src_ips": len(src_ips),
            "unique_dst_ips": len(dst_ips),
        }

    # ───── internals ─────

    def _extract_flows(self, path: str) -> dict[tuple, _FlowAcc]:
        """Read packets in batches and group into flows."""
        flows: dict[tuple, _FlowAcc] = {}

        try:
            reader = PcapReader(path)
        except Exception:
            # Fallback: load entire file at once
            logger.debug("PcapReader failed, falling back to rdpcap")
            packets = rdpcap(path)
            for pkt in packets:
                self._classify_packet(pkt, flows)
            return flows

        batch: list = []
        try:
            for pkt in reader:
                batch.append(pkt)
                if len(batch) >= self.batch_size:
                    for p in batch:
                        self._classify_packet(p, flows)
                    batch.clear()
            # Remaining
            for p in batch:
                self._classify_packet(p, flows)
        finally:
            reader.close()

        return flows

    def _classify_packet(self, pkt, flows: dict) -> None:
        """Classify one packet into the right flow accumulator."""
        if not (pkt.haslayer(IP) or pkt.haslayer(IPv6)):
            return

        ip = pkt[IP] if pkt.haslayer(IP) else pkt[IPv6]
        src_ip = ip.src
        dst_ip = ip.dst
        proto_num = int(ip.proto) if hasattr(ip, "proto") else 0
        proto_str = _proto_str(proto_num)

        src_port = _port(pkt, TCP, "sport") or _port(pkt, UDP, "sport")
        dst_port = _port(pkt, TCP, "dport") or _port(pkt, UDP, "dport")

        fwd_key = (src_ip, dst_ip, src_port, dst_port, proto_str)
        bwd_key = (dst_ip, src_ip, dst_port, src_port, proto_str)

        ts = float(pkt.time)
        pkt_len = len(pkt)

        if fwd_key in flows:
            acc = flows[fwd_key]
            is_fwd = True
        elif bwd_key in flows:
            acc = flows[bwd_key]
            is_fwd = False
        else:
            acc = _FlowAcc(
                src_ip=src_ip, dst_ip=dst_ip,
                src_port=src_port, dst_port=dst_port,
                protocol=proto_str, start_ts=ts, end_ts=ts,
            )
            flows[fwd_key] = acc
            is_fwd = True

        if ts < acc.start_ts:
            acc.start_ts = ts
        if ts > acc.end_ts:
            acc.end_ts = ts
        acc.all_ts.append(ts)

        if is_fwd:
            acc.fwd_pkts += 1
            acc.fwd_bytes += pkt_len
            acc.fwd_sizes.append(pkt_len)
            acc.fwd_ts.append(ts)
        else:
            acc.bwd_pkts += 1
            acc.bwd_bytes += pkt_len
            acc.bwd_sizes.append(pkt_len)
            acc.bwd_ts.append(ts)

        # TCP flags
        if pkt.haslayer(TCP):
            flags = pkt[TCP].flags
            if flags & 0x01: acc.fin += 1
            if flags & 0x02: acc.syn += 1
            if flags & 0x04: acc.rst += 1
            if flags & 0x08: acc.psh += 1
            if flags & 0x10: acc.ack += 1
            if flags & 0x20: acc.urg += 1
            if flags & 0x40: acc.ece += 1
            if flags & 0x80: acc.cwe += 1

            if is_fwd and acc.init_win_fwd == -1:
                acc.init_win_fwd = pkt[TCP].window
            elif not is_fwd and acc.init_win_bwd == -1:
                acc.init_win_bwd = pkt[TCP].window

        # Protocol markers
        if pkt.haslayer(DNS):
            acc.has_dns = True
        # Rough HTTP check (port 80/443 or payload starts with HTTP method)
        if dst_port in (80, 8080) or src_port in (80, 8080):
            acc.has_http = True

    # ───── feature computation ─────

    @staticmethod
    def _iat(ts_list: list) -> tuple[float, float, float, float]:
        if len(ts_list) < 2:
            return (0.0, 0.0, 0.0, 0.0)
        s = sorted(ts_list)
        iats = [s[i] - s[i - 1] for i in range(1, len(s))]
        return (
            statistics.mean(iats),
            statistics.pstdev(iats) if len(iats) > 1 else 0.0,
            max(iats),
            min(iats),
        )

    @staticmethod
    def _len_stats(sizes: list) -> tuple[float, float, float, float]:
        if not sizes:
            return (0.0, 0.0, 0.0, 0.0)
        return (
            float(max(sizes)),
            float(min(sizes)),
            statistics.mean(sizes),
            statistics.pstdev(sizes) if len(sizes) > 1 else 0.0,
        )

    def _flow_to_features(self, acc: _FlowAcc) -> dict:
        """Build a FEATURE_SCHEMA-compatible dict from a flow accumulator."""
        duration = max(acc.end_ts - acc.start_ts, 1e-6)
        total_pkts = acc.fwd_pkts + acc.bwd_pkts
        total_bytes = acc.fwd_bytes + acc.bwd_bytes
        all_sizes = acc.fwd_sizes + acc.bwd_sizes

        fwd_max, fwd_min, fwd_mean, fwd_std = self._len_stats(acc.fwd_sizes)
        bwd_max, bwd_min, bwd_mean, bwd_std = self._len_stats(acc.bwd_sizes)
        all_max, all_min, all_mean, all_std = self._len_stats(all_sizes)

        flow_iat = self._iat(acc.all_ts)
        fwd_iat = self._iat(acc.fwd_ts)
        bwd_iat = self._iat(acc.bwd_ts)

        proto_map = {"TCP": 6, "UDP": 17, "ICMP": 1}
        proto_val = proto_map.get(acc.protocol, 0)
        avg_pkt = total_bytes / total_pkts if total_pkts else 0.0
        pkt_var = statistics.pvariance(all_sizes) if all_sizes else 0.0

        features: dict = {
            "flow_duration": duration,
            "protocol_type": proto_val,
            "src_port": acc.src_port,
            "dst_port": acc.dst_port,
            "fwd_packets": acc.fwd_pkts,
            "bwd_packets": acc.bwd_pkts,
            "total_fwd_bytes": acc.fwd_bytes,
            "total_bwd_bytes": acc.bwd_bytes,
            "fwd_pkt_len_max": fwd_max,
            "fwd_pkt_len_min": fwd_min,
            "fwd_pkt_len_mean": fwd_mean,
            "fwd_pkt_len_std": fwd_std,
            "bwd_pkt_len_max": bwd_max,
            "bwd_pkt_len_min": bwd_min,
            "bwd_pkt_len_mean": bwd_mean,
            "bwd_pkt_len_std": bwd_std,
            "flow_bytes_per_sec": total_bytes / duration,
            "flow_pkts_per_sec": total_pkts / duration,
            "flow_iat_mean": flow_iat[0],
            "flow_iat_std": flow_iat[1],
            "flow_iat_max": flow_iat[2],
            "flow_iat_min": flow_iat[3],
            "fwd_iat_mean": fwd_iat[0],
            "fwd_iat_std": fwd_iat[1],
            "fwd_iat_max": fwd_iat[2],
            "fwd_iat_min": fwd_iat[3],
            "bwd_iat_mean": bwd_iat[0],
            "bwd_iat_std": bwd_iat[1],
            "bwd_iat_max": bwd_iat[2],
            "bwd_iat_min": bwd_iat[3],
            "fwd_psh_flags": 0,
            "bwd_psh_flags": 0,
            "fwd_urg_flags": 0,
            "bwd_urg_flags": 0,
            "fin_flag_cnt": acc.fin,
            "syn_flag_cnt": acc.syn,
            "rst_flag_cnt": acc.rst,
            "psh_flag_cnt": acc.psh,
            "ack_flag_cnt": acc.ack,
            "urg_flag_cnt": acc.urg,
            "ece_flag_cnt": acc.ece,
            "cwe_flag_cnt": acc.cwe,
            "pkt_len_mean": all_mean,
            "pkt_len_std": all_std,
            "pkt_len_var": pkt_var,
            "avg_pkt_size": avg_pkt,
            "fwd_seg_size_avg": fwd_mean,
            "bwd_seg_size_avg": bwd_mean,
            "subflow_fwd_pkts": acc.fwd_pkts,
            "subflow_bwd_pkts": acc.bwd_pkts,
            "subflow_fwd_bytes": acc.fwd_bytes,
            "subflow_bwd_bytes": acc.bwd_bytes,
            "init_win_bytes_fwd": acc.init_win_fwd if acc.init_win_fwd != -1 else 0,
            "init_win_bytes_bwd": acc.init_win_bwd if acc.init_win_bwd != -1 else 0,
            "act_data_pkt_fwd": acc.fwd_pkts,
            "min_seg_size_fwd": int(fwd_min) if fwd_min else 0,
            "active_mean": duration,
            "active_std": 0.0,
            "active_max": duration,
            "active_min": duration,
            "idle_mean": 0.0,
            "idle_std": 0.0,
            "idle_max": 0.0,
            "idle_min": 0.0,
            "conn_frequency": 1.0,
        }

        # Metadata (not in FEATURE_SCHEMA)
        features["_src_ip"] = acc.src_ip
        features["_dst_ip"] = acc.dst_ip
        features["_protocol"] = acc.protocol
        features["_has_dns"] = acc.has_dns
        features["_has_http"] = acc.has_http
        features["_timestamp"] = datetime.datetime.fromtimestamp(
            acc.start_ts, tz=datetime.timezone.utc
        ).isoformat()

        return features
