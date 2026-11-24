"""
Firewall rule management via Linux nftables.

Provides IP blocking / unblocking through nftables commands.
Defaults to **simulation mode** — no live firewall changes are made
unless explicitly enabled on a supported Linux host with root access.

Security invariants
-------------------
* subprocess is ALWAYS called with ``shell=False`` and an argument list.
* Every IP is validated through ``ipaddress.ip_address()`` before use.
* Protected IPs (loopback, configured management IPs) are never blocked.
* Duplicate rules are prevented.
* A configurable maximum caps the number of active blocks.
"""

from __future__ import annotations

import datetime
import ipaddress
import logging
import platform
import subprocess
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from backend.core.config import MitigationConfig, get_config
from backend.database.models import MitigationAction, SessionLocal, utcnow

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class BlockRecord:
    """In-memory record of an active block."""

    ip: str
    created_at: datetime.datetime
    expires_at: Optional[datetime.datetime]
    nftables_handle: Optional[str] = None
    is_simulated: bool = True
    reason: str = ""
    incident_id: Optional[str] = None
    db_id: Optional[str] = None


# ---------------------------------------------------------------------------
# NftablesManager
# ---------------------------------------------------------------------------

class NftablesManager:
    """Manage nftables firewall rules for IP blocking.

    Parameters
    ----------
    config : MitigationConfig, optional
        If *None*, pulled from the global ``get_config().mitigation``.
    table_name : str
        nftables table name used for managed rules.
    chain_name : str
        nftables chain name inside *table_name*.
    """

    # Class-level constants
    _DEFAULT_TABLE = "nids_block"
    _DEFAULT_CHAIN = "input_drop"

    def __init__(
        self,
        config: Optional[MitigationConfig] = None,
        table_name: str = _DEFAULT_TABLE,
        chain_name: str = _DEFAULT_CHAIN,
    ) -> None:
        self._config: MitigationConfig = config or get_config().mitigation
        self._table = table_name
        self._chain = chain_name

        # In-memory block registry  {ip_str: BlockRecord}
        self._active_blocks: Dict[str, BlockRecord] = {}

        # Determine execution mode
        self._is_linux = platform.system().lower() == "linux"
        self._simulation: bool = self._resolve_simulation_mode()

        if self._simulation:
            logger.info("NftablesManager running in SIMULATION mode")
        else:
            logger.info("NftablesManager running in LIVE mode")

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def simulation_mode(self) -> bool:
        """Return *True* when operating in simulation mode."""
        return self._simulation

    @property
    def active_block_count(self) -> int:
        """Number of currently active blocks (in-memory)."""
        return len(self._active_blocks)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def block_ip(
        self,
        ip: str,
        reason: str = "",
        duration_minutes: Optional[int] = None,
        incident_id: Optional[str] = None,
        approved_by: Optional[str] = None,
    ) -> dict:
        """Block an IP address via nftables.

        Parameters
        ----------
        ip : str
            IPv4 or IPv6 address to block.
        reason : str
            Human-readable justification.
        duration_minutes : int, optional
            Block duration. Falls back to ``MitigationConfig.default_block_duration_minutes``.
        incident_id : str, optional
            Related ``DetectionIncident.id``.
        approved_by : str, optional
            User / process that approved the block.

        Returns
        -------
        dict
            ``{"success": bool, "message": str, "simulated": bool, "ip": str}``
        """
        # --- Validate IP ---
        validated = self._validate_ip(ip)
        if validated is None:
            msg = f"Invalid IP address: {ip!r}"
            logger.warning(msg)
            self._persist_action("block", ip, reason, incident_id, approved_by, False, msg)
            return {"success": False, "message": msg, "simulated": self._simulation, "ip": ip}

        ip_str = str(validated)

        # --- Check protected ---
        if self._is_protected(ip_str):
            msg = f"Refusing to block protected IP {ip_str}"
            logger.warning(msg)
            self._persist_action("block", ip_str, reason, incident_id, approved_by, False, msg)
            return {"success": False, "message": msg, "simulated": self._simulation, "ip": ip_str}

        # --- Duplicate check ---
        if self.check_block_exists(ip_str):
            msg = f"IP {ip_str} is already blocked"
            logger.info(msg)
            return {"success": False, "message": msg, "simulated": self._simulation, "ip": ip_str}

        # --- Max blocks check ---
        if self.active_block_count >= self._config.max_active_blocks:
            msg = (
                f"Maximum active blocks ({self._config.max_active_blocks}) reached. "
                f"Cannot block {ip_str}"
            )
            logger.warning(msg)
            self._persist_action("block", ip_str, reason, incident_id, approved_by, False, msg)
            return {"success": False, "message": msg, "simulated": self._simulation, "ip": ip_str}

        # --- Compute expiry ---
        dur = duration_minutes or self._config.default_block_duration_minutes
        now = utcnow()
        expires_at = now + datetime.timedelta(minutes=dur)

        # --- Build nftables command ---
        nft_cmd = self._build_block_cmd(ip_str)
        handle: Optional[str] = None

        if self._simulation:
            logger.info("[SIMULATION] Would run: %s", " ".join(nft_cmd))
        else:
            result = self._run_nft(nft_cmd)
            if not result["success"]:
                self._persist_action(
                    "block", ip_str, reason, incident_id, approved_by, False, result["stderr"]
                )
                return {
                    "success": False,
                    "message": f"nftables command failed: {result['stderr']}",
                    "simulated": False,
                    "ip": ip_str,
                }
            handle = self._parse_handle(result.get("stdout", ""))

        # --- Register block ---
        record = BlockRecord(
            ip=ip_str,
            created_at=now,
            expires_at=expires_at,
            nftables_handle=handle,
            is_simulated=self._simulation,
            reason=reason,
            incident_id=incident_id,
        )
        self._active_blocks[ip_str] = record

        # --- Persist to DB ---
        db_id = self._persist_action(
            action_type="block",
            target_ip=ip_str,
            reason=reason,
            incident_id=incident_id,
            approved_by=approved_by,
            success=True,
            error_message=None,
            duration_minutes=dur,
            expires_at=expires_at,
            nftables_handle=handle,
            is_simulated=self._simulation,
        )
        record.db_id = db_id

        msg = (
            f"{'[SIMULATION] ' if self._simulation else ''}Blocked {ip_str} "
            f"for {dur} min (expires {expires_at.isoformat()})"
        )
        logger.info(msg)
        return {"success": True, "message": msg, "simulated": self._simulation, "ip": ip_str}

    def unblock_ip(self, ip: str, reason: str = "", approved_by: Optional[str] = None) -> dict:
        """Remove an active block for *ip*.

        Returns
        -------
        dict
            ``{"success": bool, "message": str, "simulated": bool, "ip": str}``
        """
        validated = self._validate_ip(ip)
        if validated is None:
            msg = f"Invalid IP address: {ip!r}"
            logger.warning(msg)
            return {"success": False, "message": msg, "simulated": self._simulation, "ip": ip}

        ip_str = str(validated)

        record = self._active_blocks.get(ip_str)
        if record is None:
            msg = f"No active block found for {ip_str}"
            logger.info(msg)
            return {"success": False, "message": msg, "simulated": self._simulation, "ip": ip_str}

        # --- Remove from nftables ---
        if self._simulation:
            nft_cmd = self._build_unblock_cmd(ip_str, record.nftables_handle)
            logger.info("[SIMULATION] Would run: %s", " ".join(nft_cmd))
        else:
            nft_cmd = self._build_unblock_cmd(ip_str, record.nftables_handle)
            result = self._run_nft(nft_cmd)
            if not result["success"]:
                self._persist_action(
                    "unblock", ip_str, reason, record.incident_id, approved_by, False, result["stderr"]
                )
                return {
                    "success": False,
                    "message": f"nftables unblock failed: {result['stderr']}",
                    "simulated": False,
                    "ip": ip_str,
                }

        # --- De-register ---
        del self._active_blocks[ip_str]

        # --- Mark DB record as removed ---
        self._mark_removed(record.db_id)
        self._persist_action(
            "unblock", ip_str, reason, record.incident_id, approved_by, True, None,
            is_simulated=self._simulation,
        )

        msg = f"{'[SIMULATION] ' if self._simulation else ''}Unblocked {ip_str}"
        logger.info(msg)
        return {"success": True, "message": msg, "simulated": self._simulation, "ip": ip_str}

    def list_active_blocks(self) -> List[dict]:
        """Return a list of all active blocks.

        Returns
        -------
        list[dict]
            Each entry has keys: ``ip``, ``created_at``, ``expires_at``,
            ``reason``, ``is_simulated``, ``incident_id``.
        """
        self._expire_stale_blocks()
        return [
            {
                "ip": rec.ip,
                "created_at": rec.created_at.isoformat(),
                "expires_at": rec.expires_at.isoformat() if rec.expires_at else None,
                "reason": rec.reason,
                "is_simulated": rec.is_simulated,
                "incident_id": rec.incident_id,
                "nftables_handle": rec.nftables_handle,
            }
            for rec in self._active_blocks.values()
        ]

    def check_block_exists(self, ip: str) -> bool:
        """Return *True* if *ip* is currently blocked."""
        validated = self._validate_ip(ip)
        if validated is None:
            return False
        self._expire_stale_blocks()
        return str(validated) in self._active_blocks

    def expire_blocks(self) -> List[str]:
        """Expire any blocks whose duration has elapsed.

        Returns
        -------
        list[str]
            IPs that were unblocked due to expiry.
        """
        return self._expire_stale_blocks()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _resolve_simulation_mode(self) -> bool:
        """Decide whether to simulate.

        * Config says simulation → simulate.
        * Not on Linux → always simulate.
        * On Linux but not root → simulate and warn.
        """
        if self._config.simulation_mode:
            return True
        if not self._is_linux:
            logger.warning(
                "Not on Linux — forcing simulation mode regardless of config"
            )
            return True
        # On Linux, check for root
        try:
            import os

            if os.geteuid() != 0:
                logger.warning(
                    "Not running as root — forcing simulation mode"
                )
                return True
        except AttributeError:
            # os.geteuid() not available (shouldn't happen on Linux)
            return True
        return False

    # -- IP helpers --

    @staticmethod
    def _validate_ip(ip: str) -> Optional[ipaddress.IPv4Address | ipaddress.IPv6Address]:
        """Parse and return a valid IP, or *None*."""
        try:
            return ipaddress.ip_address(ip.strip())
        except (ValueError, AttributeError):
            return None

    def _is_protected(self, ip_str: str) -> bool:
        """Check whether *ip_str* is a protected IP."""
        protected = set(self._config.protected_ips)
        # Always protect loopback
        protected.update({"127.0.0.1", "::1"})
        return ip_str in protected

    # -- nftables command builders --

    def _build_block_cmd(self, ip_str: str) -> list:
        """Build an nftables *add rule* command list (no shell)."""
        return [
            "nft",
            "add", "rule",
            "inet", self._table, self._chain,
            "ip", "saddr", ip_str,
            "drop",
        ]

    def _build_unblock_cmd(self, ip_str: str, handle: Optional[str]) -> list:
        """Build an nftables *delete rule* command list."""
        if handle:
            return [
                "nft",
                "delete", "rule",
                "inet", self._table, self._chain,
                "handle", handle,
            ]
        # Fallback: flush the whole chain (less precise, but safe)
        logger.warning(
            "No nftables handle for %s — building flush command for the managed chain", ip_str
        )
        return [
            "nft",
            "flush", "chain",
            "inet", self._table, self._chain,
        ]

    # -- subprocess runner --

    @staticmethod
    def _run_nft(cmd: list) -> dict:
        """Run an nftables command safely.

        Returns
        -------
        dict
            ``{"success": bool, "stdout": str, "stderr": str, "returncode": int}``
        """
        try:
            proc = subprocess.run(
                cmd,
                shell=False,
                capture_output=True,
                text=True,
                timeout=10,
            )
            return {
                "success": proc.returncode == 0,
                "stdout": proc.stdout.strip(),
                "stderr": proc.stderr.strip(),
                "returncode": proc.returncode,
            }
        except FileNotFoundError:
            return {
                "success": False,
                "stdout": "",
                "stderr": "nft binary not found",
                "returncode": -1,
            }
        except subprocess.TimeoutExpired:
            return {
                "success": False,
                "stdout": "",
                "stderr": "nft command timed out",
                "returncode": -1,
            }
        except Exception as exc:  # noqa: BLE001
            return {
                "success": False,
                "stdout": "",
                "stderr": str(exc),
                "returncode": -1,
            }

    @staticmethod
    def _parse_handle(stdout: str) -> Optional[str]:
        """Extract the nftables handle number from command output."""
        # nft with ``-e --handle`` often returns "# handle <N>"
        for token in reversed(stdout.split()):
            if token.isdigit():
                return token
        return None

    # -- Expiry --

    def _expire_stale_blocks(self) -> List[str]:
        """Remove blocks whose ``expires_at`` is in the past."""
        now = utcnow()
        expired: List[str] = []
        for ip_str, rec in list(self._active_blocks.items()):
            if rec.expires_at and rec.expires_at <= now:
                logger.info("Block on %s expired", ip_str)
                # Attempt to remove live rule
                if not rec.is_simulated and rec.nftables_handle:
                    self._run_nft(self._build_unblock_cmd(ip_str, rec.nftables_handle))
                self._mark_removed(rec.db_id, status="expired")
                del self._active_blocks[ip_str]
                expired.append(ip_str)
        return expired

    # -- Database persistence --

    def _persist_action(
        self,
        action_type: str,
        target_ip: str,
        reason: str = "",
        incident_id: Optional[str] = None,
        approved_by: Optional[str] = None,
        success: bool = True,
        error_message: Optional[str] = None,
        duration_minutes: Optional[int] = None,
        expires_at: Optional[datetime.datetime] = None,
        nftables_handle: Optional[str] = None,
        is_simulated: bool = True,
    ) -> Optional[str]:
        """Write a ``MitigationAction`` to the database.

        Returns the ``id`` of the created row, or *None* on failure.
        """
        try:
            session = SessionLocal()
            action = MitigationAction(
                action_type=action_type,
                target_ip=target_ip,
                reason=reason,
                incident_id=incident_id,
                duration_minutes=duration_minutes,
                expires_at=expires_at,
                is_active=success and action_type == "block",
                is_simulated=is_simulated,
                approved_by=approved_by,
                nftables_handle=nftables_handle,
                status="active" if (success and action_type == "block") else (
                    "removed" if action_type == "unblock" else "failed"
                ),
                error_message=error_message,
            )
            session.add(action)
            session.commit()
            action_id = action.id
            session.close()
            return action_id
        except Exception:
            logger.exception("Failed to persist MitigationAction")
            return None

    @staticmethod
    def _mark_removed(db_id: Optional[str], status: str = "removed") -> None:
        """Update a ``MitigationAction`` row to removed / expired."""
        if not db_id:
            return
        try:
            session = SessionLocal()
            row = session.query(MitigationAction).filter_by(id=db_id).first()
            if row:
                row.is_active = False
                row.status = status
                row.removed_at = utcnow()
                session.commit()
            session.close()
        except Exception:
            logger.exception("Failed to update MitigationAction %s", db_id)
