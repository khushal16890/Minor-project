"""
Automated response coordinator.

Maps risk scores to response levels (LOW → CRITICAL), coordinates
blocking decisions, tracks repeated activity from the same source,
and maintains a rollback mechanism for every block it creates.

Safety constraints from :class:`MitigationConfig` are always respected:

* ``enabled`` must be ``True`` for any real action.
* ``auto_block_critical`` must be ``True`` for unattended blocking.
* ``simulation_mode`` is forwarded to :class:`NftablesManager`.
* Protected IPs and management interface are never touched.
"""

from __future__ import annotations

import logging
from collections import defaultdict
from dataclasses import dataclass, field
from enum import IntEnum
from typing import Dict, List, Optional

from backend.core.config import MitigationConfig, get_config
from backend.database.models import (
    DetectionIncident,
    MitigationAction,
    SessionLocal,
    utcnow,
)
from mitigation.audit import AuditLogger
from mitigation.nftables import NftablesManager

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Response levels
# ---------------------------------------------------------------------------

class ResponseLevel(IntEnum):
    """Graduated response levels."""

    LOW = 0
    MEDIUM = 1
    HIGH = 2
    CRITICAL = 3


# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------

@dataclass
class SourceActivity:
    """Track repeated activity from one source IP."""

    ip: str
    hit_count: int = 0
    last_seen: Optional[str] = None  # ISO timestamp
    highest_risk: float = 0.0
    highest_level: ResponseLevel = ResponseLevel.LOW
    incident_ids: List[str] = field(default_factory=list)


@dataclass
class RollbackEntry:
    """Record of a block that can be undone."""

    ip: str
    block_result: dict
    incident_id: Optional[str] = None
    response_level: ResponseLevel = ResponseLevel.CRITICAL
    reason: str = ""


# ---------------------------------------------------------------------------
# ResponseEngine
# ---------------------------------------------------------------------------

class ResponseEngine:
    """Automated response coordinator.

    Parameters
    ----------
    config : MitigationConfig, optional
        Falls back to global config if not supplied.
    nft_manager : NftablesManager, optional
        Injected firewall manager; created from *config* if absent.
    audit_logger : AuditLogger, optional
        Injected audit logger; created if absent.
    """

    # Risk-score → response-level boundaries (from config thresholds)
    _DEFAULT_BOUNDARIES = {
        "low_max": 29,
        "medium_max": 59,
        "high_max": 79,
    }

    def __init__(
        self,
        config: Optional[MitigationConfig] = None,
        nft_manager: Optional[NftablesManager] = None,
        audit_logger: Optional[AuditLogger] = None,
    ) -> None:
        self._config: MitigationConfig = config or get_config().mitigation
        self._nft: NftablesManager = nft_manager or NftablesManager(config=self._config)
        self._audit: AuditLogger = audit_logger or AuditLogger()

        # {ip_str: SourceActivity}
        self._source_tracker: Dict[str, SourceActivity] = defaultdict(
            lambda: SourceActivity(ip="")
        )

        # Rollback stack — newest first
        self._rollback_stack: List[RollbackEntry] = []

        # Load risk thresholds
        app_cfg = get_config()
        thresholds = app_cfg.risk_thresholds
        self._low_max: int = thresholds.low_max
        self._medium_max: int = thresholds.medium_max
        self._high_max: int = thresholds.high_max

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def evaluate_and_respond(
        self,
        risk_score: float,
        src_ip: str,
        incident_id: Optional[str] = None,
        attack_type: Optional[str] = None,
        evidence: Optional[dict] = None,
        approved_by: Optional[str] = None,
    ) -> dict:
        """Evaluate a risk score and take the appropriate response.

        Parameters
        ----------
        risk_score : float
            Composite risk score (0–100).
        src_ip : str
            Source IP that generated the event.
        incident_id : str, optional
            Related ``DetectionIncident.id``.
        attack_type : str, optional
            Detected attack category.
        evidence : dict, optional
            Supporting evidence payload.
        approved_by : str, optional
            Admin who approved (required for HIGH; optional for CRITICAL auto-block).

        Returns
        -------
        dict
            ``{"level": str, "action_taken": str, "details": dict}``
        """
        level = self.score_to_level(risk_score)

        # Track source activity
        self._track_source(src_ip, risk_score, level, incident_id)

        detail = {
            "risk_score": risk_score,
            "src_ip": src_ip,
            "incident_id": incident_id,
            "attack_type": attack_type,
            "response_level": level.name,
            "evidence": evidence,
        }

        if level == ResponseLevel.LOW:
            return self._handle_low(detail)
        elif level == ResponseLevel.MEDIUM:
            return self._handle_medium(detail)
        elif level == ResponseLevel.HIGH:
            return self._handle_high(detail, approved_by)
        else:  # CRITICAL
            return self._handle_critical(detail, approved_by)

    def score_to_level(self, risk_score: float) -> ResponseLevel:
        """Map a numeric risk score to a :class:`ResponseLevel`.

        Parameters
        ----------
        risk_score : float
            Score between 0 and 100.

        Returns
        -------
        ResponseLevel
        """
        score = max(0.0, min(100.0, risk_score))
        if score <= self._low_max:
            return ResponseLevel.LOW
        if score <= self._medium_max:
            return ResponseLevel.MEDIUM
        if score <= self._high_max:
            return ResponseLevel.HIGH
        return ResponseLevel.CRITICAL

    def get_source_activity(self, ip: str) -> Optional[dict]:
        """Return activity summary for *ip*, or *None* if unseen.

        Returns
        -------
        dict or None
            Keys: ``ip``, ``hit_count``, ``last_seen``, ``highest_risk``,
            ``highest_level``, ``incident_ids``.
        """
        activity = self._source_tracker.get(ip)
        if activity is None or activity.hit_count == 0:
            return None
        return {
            "ip": activity.ip,
            "hit_count": activity.hit_count,
            "last_seen": activity.last_seen,
            "highest_risk": activity.highest_risk,
            "highest_level": activity.highest_level.name,
            "incident_ids": list(activity.incident_ids),
        }

    def rollback_last(self) -> dict:
        """Undo the most recent block by popping the rollback stack.

        Returns
        -------
        dict
            ``{"success": bool, "message": str, "ip": str | None}``
        """
        if not self._rollback_stack:
            return {"success": False, "message": "Rollback stack is empty", "ip": None}

        entry = self._rollback_stack.pop()
        result = self._nft.unblock_ip(
            ip=entry.ip,
            reason=f"Rollback: {entry.reason}",
            approved_by="rollback_engine",
        )
        self._audit.log_mitigation(
            action="rollback_unblock",
            details={
                "ip": entry.ip,
                "incident_id": entry.incident_id,
                "unblock_result": result,
            },
            user="rollback_engine",
            ip_address=entry.ip,
            success=result["success"],
        )
        logger.info("Rollback unblock %s: %s", entry.ip, result["message"])
        return {
            "success": result["success"],
            "message": result["message"],
            "ip": entry.ip,
        }

    def rollback_all(self) -> List[dict]:
        """Undo every block on the rollback stack.

        Returns
        -------
        list[dict]
            One result per rollback attempt.
        """
        results: List[dict] = []
        while self._rollback_stack:
            results.append(self.rollback_last())
        return results

    def get_rollback_stack(self) -> List[dict]:
        """Return the current rollback stack (newest first)."""
        return [
            {
                "ip": e.ip,
                "incident_id": e.incident_id,
                "response_level": e.response_level.name,
                "reason": e.reason,
            }
            for e in self._rollback_stack
        ]

    # ------------------------------------------------------------------
    # Level handlers
    # ------------------------------------------------------------------

    def _handle_low(self, detail: dict) -> dict:
        """LOW: record event only."""
        self._audit.log_detection(
            action="response_low",
            details=detail,
            user="response_engine",
        )
        logger.debug("LOW response for %s (score %.1f)", detail["src_ip"], detail["risk_score"])
        return {
            "level": "LOW",
            "action_taken": "recorded",
            "details": detail,
        }

    def _handle_medium(self, detail: dict) -> dict:
        """MEDIUM: generate alert, monitor for repeated activity."""
        activity = self._source_tracker.get(detail["src_ip"])
        repeated = activity is not None and activity.hit_count > 1

        self._audit.log_detection(
            action="response_medium",
            details={**detail, "repeated": repeated, "hit_count": activity.hit_count if activity else 0},
            user="response_engine",
        )

        action_desc = "alert_generated"
        if repeated:
            action_desc = "alert_generated_repeated_activity"
            logger.warning(
                "MEDIUM response with repeated activity for %s (hits=%d)",
                detail["src_ip"],
                activity.hit_count if activity else 0,
            )
        else:
            logger.info("MEDIUM response for %s (score %.1f)", detail["src_ip"], detail["risk_score"])

        return {
            "level": "MEDIUM",
            "action_taken": action_desc,
            "details": {**detail, "repeated": repeated},
        }

    def _handle_high(self, detail: dict, approved_by: Optional[str]) -> dict:
        """HIGH: require administrator approval before blocking."""
        if not self._config.enabled:
            self._audit.log_mitigation(
                action="response_high_disabled",
                details=detail,
                user="response_engine",
                success=False,
            )
            return {
                "level": "HIGH",
                "action_taken": "mitigation_disabled",
                "details": {**detail, "note": "Mitigation engine is disabled in config"},
            }

        if not approved_by:
            self._audit.log_mitigation(
                action="response_high_pending_approval",
                details=detail,
                user="response_engine",
            )
            logger.warning(
                "HIGH response for %s requires admin approval (score %.1f)",
                detail["src_ip"], detail["risk_score"],
            )
            return {
                "level": "HIGH",
                "action_taken": "pending_approval",
                "details": {**detail, "note": "Administrator approval required to block"},
            }

        # Approved → block
        return self._execute_block(detail, approved_by, ResponseLevel.HIGH)

    def _handle_critical(self, detail: dict, approved_by: Optional[str]) -> dict:
        """CRITICAL: auto-block ONLY when explicitly enabled AND safe."""
        if not self._config.enabled:
            self._audit.log_mitigation(
                action="response_critical_disabled",
                details=detail,
                user="response_engine",
                success=False,
            )
            return {
                "level": "CRITICAL",
                "action_taken": "mitigation_disabled",
                "details": {**detail, "note": "Mitigation engine is disabled in config"},
            }

        # If admin has approved, block immediately
        if approved_by:
            return self._execute_block(detail, approved_by, ResponseLevel.CRITICAL)

        # Auto-block path: needs auto_block_critical AND safety conditions
        if not self._config.auto_block_critical:
            self._audit.log_mitigation(
                action="response_critical_auto_block_disabled",
                details=detail,
                user="response_engine",
            )
            logger.warning(
                "CRITICAL response for %s — auto-block disabled (score %.1f)",
                detail["src_ip"], detail["risk_score"],
            )
            return {
                "level": "CRITICAL",
                "action_taken": "pending_approval",
                "details": {
                    **detail,
                    "note": "Auto-block is disabled; administrator approval required",
                },
            }

        # Safety conditions: must not be protected IP, must have capacity
        if not self._safety_check(detail["src_ip"]):
            self._audit.log_mitigation(
                action="response_critical_safety_failed",
                details=detail,
                user="response_engine",
                success=False,
            )
            return {
                "level": "CRITICAL",
                "action_taken": "safety_check_failed",
                "details": {
                    **detail,
                    "note": "Safety conditions not met for auto-block",
                },
            }

        # All checks passed — auto-block
        return self._execute_block(detail, "auto_block", ResponseLevel.CRITICAL)

    # ------------------------------------------------------------------
    # Blocking helper
    # ------------------------------------------------------------------

    def _execute_block(self, detail: dict, approved_by: str, level: ResponseLevel) -> dict:
        """Run the actual block via :class:`NftablesManager` and log it.

        Returns
        -------
        dict
            Standard response dict.
        """
        reason = (
            f"{level.name} response — risk {detail['risk_score']:.1f}, "
            f"attack={detail.get('attack_type', 'unknown')}"
        )
        block_result = self._nft.block_ip(
            ip=detail["src_ip"],
            reason=reason,
            incident_id=detail.get("incident_id"),
            approved_by=approved_by,
        )

        self._audit.log_mitigation(
            action=f"response_{level.name.lower()}_block",
            details={
                **detail,
                "block_result": block_result,
                "approved_by": approved_by,
            },
            user=approved_by,
            ip_address=detail["src_ip"],
            success=block_result["success"],
        )

        # Push to rollback stack on success
        if block_result["success"]:
            self._rollback_stack.append(
                RollbackEntry(
                    ip=detail["src_ip"],
                    block_result=block_result,
                    incident_id=detail.get("incident_id"),
                    response_level=level,
                    reason=reason,
                )
            )

        action_taken = "blocked" if block_result["success"] else "block_failed"
        if block_result.get("simulated"):
            action_taken = "blocked_simulated"

        return {
            "level": level.name,
            "action_taken": action_taken,
            "details": {**detail, "block_result": block_result},
        }

    # ------------------------------------------------------------------
    # Source tracking
    # ------------------------------------------------------------------

    def _track_source(
        self,
        ip: str,
        risk_score: float,
        level: ResponseLevel,
        incident_id: Optional[str],
    ) -> None:
        """Update the activity tracker for *ip*."""
        activity = self._source_tracker[ip]
        if activity.ip == "":
            activity.ip = ip
        activity.hit_count += 1
        activity.last_seen = utcnow().isoformat()
        if risk_score > activity.highest_risk:
            activity.highest_risk = risk_score
        if level > activity.highest_level:
            activity.highest_level = level
        if incident_id and incident_id not in activity.incident_ids:
            activity.incident_ids.append(incident_id)

    # ------------------------------------------------------------------
    # Safety checks
    # ------------------------------------------------------------------

    def _safety_check(self, ip: str) -> bool:
        """Verify that it is safe to auto-block *ip*.

        Conditions checked:
        1. IP is not protected (loopback, management IPs).
        2. Active block count has not reached the maximum.
        """
        # Protected IP check is also done inside NftablesManager, but we
        # short-circuit here for a cleaner audit trail.
        protected = set(self._config.protected_ips) | {"127.0.0.1", "::1"}
        if ip in protected:
            logger.warning("Safety: %s is a protected IP", ip)
            return False
        if self._nft.active_block_count >= self._config.max_active_blocks:
            logger.warning("Safety: max active blocks reached")
            return False
        return True
