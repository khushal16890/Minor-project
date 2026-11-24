"""
Live network traffic capture module using Scapy.

Captures packets from a specified network interface, converts them into
flow-level data keyed by the standard 5-tuple (src_ip, dst_ip, src_port,
dst_port, protocol), and optionally writes captured traffic to PCAP files.

Provides a Windows-compatible fallback (uses ``conf.L3socket`` when Npcap
is detected) and supports BPF filter strings.
"""

import os
import sys
import time
import logging
import datetime
import threading
from typing import Optional, Callable
from collections import defaultdict
from dataclasses import dataclass, field

try:
    from scapy.all import (
        sniff, wrpcap, conf,
        IP, IPv6, TCP, UDP, ICMP, DNS,
    )
except ImportError:
    raise ImportError(
        "Scapy is required for traffic capture. "
        "Install it with: pip install scapy"
    )

from backend.core.config import FEATURE_SCHEMA, NetworkConfig

logger = logging.getLogger(__name__)

# ─────────────────────── helpers ───────────────────────


def _protocol_num_to_str(proto_num: int) -> str:
    """Map IP protocol number to human-readable name."""
    return {1: "ICMP", 6: "TCP", 17: "UDP"}.get(proto_num, str(proto_num))


def _safe_port(pkt, layer, attr: str) -> int:
    """Extract port from a packet layer, returning 0 when absent."""
    if pkt.haslayer(layer):
        return int(getattr(pkt[layer], attr, 0))
    return 0


# ─────────────────────── data classes ───────────────────────


@dataclass
class FlowKey:
    """Canonical 5-tuple identifying a network flow."""
    src_ip: str
    dst_ip: str
    src_port: int
    dst_port: int
    protocol: str

    def __hash__(self):
        return hash((self.src_ip, self.dst_ip, self.src_port,
                      self.dst_port, self.protocol))

    def __eq__(self, other):
        if not isinstance(other, FlowKey):
            return NotImplemented
        return (self.src_ip == other.src_ip and self.dst_ip == other.dst_ip and
                self.src_port == other.src_port and self.dst_port == other.dst_port and
                self.protocol == other.protocol)


@dataclass
class FlowRecord:
    """Accumulates per-flow statistics during live capture."""
    key: FlowKey
    start_time: float
    last_time: float
    fwd_packets: int = 0
    bwd_packets: int = 0
    total_fwd_bytes: int = 0
    total_bwd_bytes: int = 0
    fwd_pkt_sizes: list = field(default_factory=list)
    bwd_pkt_sizes: list = field(default_factory=list)
    timestamps: list = field(default_factory=list)
    fwd_timestamps: list = field(default_factory=list)
    bwd_timestamps: list = field(default_factory=list)
    # TCP flag counters
    fin_cnt: int = 0
    syn_cnt: int = 0
    rst_cnt: int = 0
    psh_cnt: int = 0
    ack_cnt: int = 0
    urg_cnt: int = 0
    ece_cnt: int = 0
    cwe_cnt: int = 0
    # Initial TCP window sizes
    init_win_fwd: int = -1
    init_win_bwd: int = -1


# ─────────────────────── TrafficCapture ───────────────────────


class TrafficCapture:
    """Live network packet capture with flow aggregation.

    Parameters
    ----------
    config : NetworkConfig, optional
        Network monitoring configuration.  Defaults are sensible for
        quick local testing.
    """

    def __init__(self, config: Optional[NetworkConfig] = None):
        self.config = config or NetworkConfig()
        self._flows: dict[FlowKey, FlowRecord] = {}
        self._lock = threading.Lock()
        self._packets: list = []
        self._running = False
        self._capture_thread: Optional[threading.Thread] = None

        # Windows compatibility: try to use Npcap L3 socket
        if sys.platform == "win32":
            try:
                conf.use_npcap = True
            except Exception:
                logger.warning("Npcap not detected – raw socket capture may fail on Windows.")

    # ───── public API ─────

    def start(
        self,
        interface: Optional[str] = None,
        bpf_filter: Optional[str] = None,
        count: int = 0,
        timeout: Optional[int] = None,
        callback: Optional[Callable] = None,
    ) -> None:
        """Begin capturing packets on a background thread.

        Parameters
        ----------
        interface : str, optional
            Network interface to listen on (e.g. ``"eth0"``).  Falls back
            to :pyattr:`config.capture_interface` or Scapy's default.
        bpf_filter : str, optional
            BPF filter string (e.g. ``"tcp port 80"``).
        count : int
            Stop after capturing *count* packets (0 = unlimited).
        timeout : int, optional
            Stop capturing after *timeout* seconds.
        callback : callable, optional
            Called with each packet *after* internal processing.
        """
        if self._running:
            logger.warning("Capture already running.")
            return

        iface = interface or self.config.capture_interface
        filt = bpf_filter or self.config.capture_filter

        self._running = True
        self._capture_thread = threading.Thread(
            target=self._capture_loop,
            kwargs={
                "iface": iface,
                "bpf_filter": filt,
                "count": count,
                "timeout": timeout,
                "user_callback": callback,
            },
            daemon=True,
        )
        self._capture_thread.start()
        logger.info("Capture started on interface=%s filter='%s'", iface, filt)

    def stop(self) -> None:
        """Signal the capture thread to stop."""
        self._running = False
        if self._capture_thread and self._capture_thread.is_alive():
            self._capture_thread.join(timeout=5)
        logger.info("Capture stopped. Packets collected: %d", len(self._packets))

    def capture_sync(
        self,
        interface: Optional[str] = None,
        bpf_filter: Optional[str] = None,
        count: int = 100,
        timeout: int = 30,
    ) -> list:
        """Blocking capture that returns raw packets.

        Convenience wrapper around Scapy ``sniff`` for one-shot use.
        """
        iface = interface or self.config.capture_interface
        filt = bpf_filter or self.config.capture_filter

        logger.info("Synchronous capture: iface=%s filter='%s' count=%d timeout=%ds",
                     iface, filt, count, timeout)

        packets = sniff(
            iface=iface,
            filter=filt,
            count=count,
            timeout=timeout,
            prn=self._process_packet,
            store=True,
        )
        return list(packets)

    def get_flows(self) -> dict[FlowKey, FlowRecord]:
        """Return a snapshot of all tracked flows."""
        with self._lock:
            return dict(self._flows)

    def get_flow_features(self) -> list[dict]:
        """Convert tracked flows into feature dicts compatible with
        ``FEATURE_SCHEMA``.

        Returns
        -------
        list[dict]
            One dict per flow, keys drawn from ``FEATURE_SCHEMA``.
        """
        with self._lock:
            flows = dict(self._flows)

        results = []
        for key, rec in flows.items():
            results.append(self._flow_to_features(key, rec))
        return results

    def write_pcap(self, path: str) -> str:
        """Write all captured packets to a PCAP file.

        Parameters
        ----------
        path : str
            Destination file path (e.g. ``"capture.pcap"``).

        Returns
        -------
        str
            Absolute path of the written file.
        """
        abs_path = os.path.abspath(path)
        os.makedirs(os.path.dirname(abs_path) or ".", exist_ok=True)
        wrpcap(abs_path, self._packets)
        logger.info("Wrote %d packets to %s", len(self._packets), abs_path)
        return abs_path

    def reset(self) -> None:
        """Clear captured packets and flow state."""
        with self._lock:
            self._flows.clear()
        self._packets.clear()

    # ───── internals ─────

    def _capture_loop(
        self,
        iface: Optional[str],
        bpf_filter: Optional[str],
        count: int,
        timeout: Optional[int],
        user_callback: Optional[Callable],
    ) -> None:
        """Background thread target."""
        try:
            sniff(
                iface=iface,
                filter=bpf_filter,
                count=count,
                timeout=timeout,
                prn=lambda pkt: self._on_packet(pkt, user_callback),
                stop_filter=lambda _: not self._running,
                store=False,
            )
        except Exception as exc:
            logger.error("Capture error: %s", exc)
        finally:
            self._running = False

    def _on_packet(self, pkt, user_callback: Optional[Callable]) -> None:
        self._process_packet(pkt)
        if user_callback:
            try:
                user_callback(pkt)
            except Exception as exc:
                logger.debug("User callback error: %s", exc)

    def _process_packet(self, pkt) -> None:
        """Extract flow information from a single packet and update state."""
        self._packets.append(pkt)

        if not (pkt.haslayer(IP) or pkt.haslayer(IPv6)):
            return

        ip_layer = pkt[IP] if pkt.haslayer(IP) else pkt[IPv6]
        src_ip = ip_layer.src
        dst_ip = ip_layer.dst
        proto_num = int(ip_layer.proto) if hasattr(ip_layer, "proto") else 0
        proto_str = _protocol_num_to_str(proto_num)

        src_port = _safe_port(pkt, TCP, "sport") or _safe_port(pkt, UDP, "sport")
        dst_port = _safe_port(pkt, TCP, "dport") or _safe_port(pkt, UDP, "dport")

        fwd_key = FlowKey(src_ip, dst_ip, src_port, dst_port, proto_str)
        bwd_key = FlowKey(dst_ip, src_ip, dst_port, src_port, proto_str)

        ts = float(pkt.time)
        pkt_len = len(pkt)

        with self._lock:
            # Determine direction
            if fwd_key in self._flows:
                flow = self._flows[fwd_key]
                is_forward = True
            elif bwd_key in self._flows:
                flow = self._flows[bwd_key]
                is_forward = False
            else:
                # New flow
                flow = FlowRecord(key=fwd_key, start_time=ts, last_time=ts)
                self._flows[fwd_key] = flow
                is_forward = True

            flow.last_time = ts
            flow.timestamps.append(ts)

            if is_forward:
                flow.fwd_packets += 1
                flow.total_fwd_bytes += pkt_len
                flow.fwd_pkt_sizes.append(pkt_len)
                flow.fwd_timestamps.append(ts)
            else:
                flow.bwd_packets += 1
                flow.total_bwd_bytes += pkt_len
                flow.bwd_pkt_sizes.append(pkt_len)
                flow.bwd_timestamps.append(ts)

            # TCP flags
            if pkt.haslayer(TCP):
                flags = pkt[TCP].flags
                if flags & 0x01:
                    flow.fin_cnt += 1
                if flags & 0x02:
                    flow.syn_cnt += 1
                if flags & 0x04:
                    flow.rst_cnt += 1
                if flags & 0x08:
                    flow.psh_cnt += 1
                if flags & 0x10:
                    flow.ack_cnt += 1
                if flags & 0x20:
                    flow.urg_cnt += 1
                if flags & 0x40:
                    flow.ece_cnt += 1
                if flags & 0x80:
                    flow.cwe_cnt += 1

                # Initial window sizes
                if is_forward and flow.init_win_fwd == -1:
                    flow.init_win_fwd = pkt[TCP].window
                elif not is_forward and flow.init_win_bwd == -1:
                    flow.init_win_bwd = pkt[TCP].window

    # ───── feature computation ─────

    @staticmethod
    def _iat_stats(timestamps: list) -> tuple[float, float, float, float]:
        """Compute inter-arrival time statistics from a sorted timestamp list."""
        if len(timestamps) < 2:
            return (0.0, 0.0, 0.0, 0.0)
        iats = [timestamps[i] - timestamps[i - 1] for i in range(1, len(timestamps))]
        import statistics
        mean_iat = statistics.mean(iats)
        std_iat = statistics.pstdev(iats) if len(iats) > 1 else 0.0
        return (mean_iat, std_iat, max(iats), min(iats))

    @staticmethod
    def _pkt_len_stats(sizes: list) -> tuple[float, float, float, float]:
        """Max, min, mean, std of packet sizes."""
        if not sizes:
            return (0.0, 0.0, 0.0, 0.0)
        import statistics
        mean_v = statistics.mean(sizes)
        std_v = statistics.pstdev(sizes) if len(sizes) > 1 else 0.0
        return (max(sizes), min(sizes), mean_v, std_v)

    def _flow_to_features(self, key: FlowKey, rec: FlowRecord) -> dict:
        """Build a feature dict matching ``FEATURE_SCHEMA`` from a FlowRecord."""
        duration = max(rec.last_time - rec.start_time, 1e-6)
        total_pkts = rec.fwd_packets + rec.bwd_packets
        total_bytes = rec.total_fwd_bytes + rec.total_bwd_bytes
        all_sizes = rec.fwd_pkt_sizes + rec.bwd_pkt_sizes

        fwd_max, fwd_min, fwd_mean, fwd_std = self._pkt_len_stats(rec.fwd_pkt_sizes)
        bwd_max, bwd_min, bwd_mean, bwd_std = self._pkt_len_stats(rec.bwd_pkt_sizes)
        all_max, all_min, all_mean, all_std = self._pkt_len_stats(all_sizes)

        flow_iat = self._iat_stats(sorted(rec.timestamps))
        fwd_iat = self._iat_stats(sorted(rec.fwd_timestamps))
        bwd_iat = self._iat_stats(sorted(rec.bwd_timestamps))

        proto_map = {"TCP": 6, "UDP": 17, "ICMP": 1}
        proto_val = proto_map.get(key.protocol, 0)

        import statistics
        all_var = statistics.pvariance(all_sizes) if all_sizes else 0.0
        avg_pkt_size = total_bytes / total_pkts if total_pkts else 0.0

        features = {
            "flow_duration": duration,
            "protocol_type": proto_val,
            "src_port": key.src_port,
            "dst_port": key.dst_port,
            "fwd_packets": rec.fwd_packets,
            "bwd_packets": rec.bwd_packets,
            "total_fwd_bytes": rec.total_fwd_bytes,
            "total_bwd_bytes": rec.total_bwd_bytes,
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
            "fin_flag_cnt": rec.fin_cnt,
            "syn_flag_cnt": rec.syn_cnt,
            "rst_flag_cnt": rec.rst_cnt,
            "psh_flag_cnt": rec.psh_cnt,
            "ack_flag_cnt": rec.ack_cnt,
            "urg_flag_cnt": rec.urg_cnt,
            "ece_flag_cnt": rec.ece_cnt,
            "cwe_flag_cnt": rec.cwe_cnt,
            "pkt_len_mean": all_mean,
            "pkt_len_std": all_std,
            "pkt_len_var": all_var,
            "avg_pkt_size": avg_pkt_size,
            "fwd_seg_size_avg": fwd_mean,
            "bwd_seg_size_avg": bwd_mean,
            "subflow_fwd_pkts": rec.fwd_packets,
            "subflow_bwd_pkts": rec.bwd_packets,
            "subflow_fwd_bytes": rec.total_fwd_bytes,
            "subflow_bwd_bytes": rec.total_bwd_bytes,
            "init_win_bytes_fwd": rec.init_win_fwd if rec.init_win_fwd != -1 else 0,
            "init_win_bytes_bwd": rec.init_win_bwd if rec.init_win_bwd != -1 else 0,
            "act_data_pkt_fwd": rec.fwd_packets,
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

        # Metadata (not in FEATURE_SCHEMA but useful downstream)
        features["_src_ip"] = key.src_ip
        features["_dst_ip"] = key.dst_ip
        features["_timestamp"] = datetime.datetime.fromtimestamp(
            rec.start_time, tz=datetime.timezone.utc
        ).isoformat()

        return features
