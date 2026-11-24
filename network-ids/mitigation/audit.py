"""
Audit logging subsystem.

Every mitigation action and significant system event is logged to the
``audit_logs`` database table via the :class:`AuditLogger` class.

Records track: timestamp, action, originating module, structured details,
user, and success / failure status.
"""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Any, Dict, List, Optional

from backend.database.models import AuditLog, SessionLocal, utcnow

logger = logging.getLogger(__name__)


class AuditLogger:
    """Centralised audit-trail writer and query interface.

    Every call to :meth:`log` creates a persistent ``AuditLog`` row in the
    database.  The class also provides convenience wrappers for common event
    categories and query helpers for retrieving the trail.

    Example
    -------
    >>> audit = AuditLogger()
    >>> audit.log_mitigation("block_ip", details={"ip": "10.0.0.5"}, user="system")
    """

    # ------------------------------------------------------------------
    # Core logging
    # ------------------------------------------------------------------

    def log(
        self,
        action: str,
        module: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
        user: Optional[str] = None,
        ip_address: Optional[str] = None,
        success: bool = True,
    ) -> Optional[str]:
        """Write an audit record to the database.

        Parameters
        ----------
        action : str
            Short description of the event (e.g. ``"block_ip"``).
        module : str, optional
            Originating module or subsystem name.
        details : dict, optional
            Structured payload with event-specific information.
        user : str, optional
            User or process identity that triggered the action.
        ip_address : str, optional
            IP address associated with the event (source or target).
        success : bool
            Whether the action completed successfully.

        Returns
        -------
        str or None
            The ``id`` of the created ``AuditLog`` row, or *None* on failure.
        """
        try:
            session = SessionLocal()
            record = AuditLog(
                action=action,
                module=module,
                details=details,
                user=user,
                ip_address=ip_address,
                success=success,
            )
            session.add(record)
            session.commit()
            record_id = record.id
            session.close()
            logger.debug(
                "Audit: action=%s module=%s success=%s user=%s",
                action, module, success, user,
            )
            return record_id
        except Exception:
            logger.exception("Failed to write audit log for action=%s", action)
            return None

    # ------------------------------------------------------------------
    # Convenience wrappers
    # ------------------------------------------------------------------

    def log_mitigation(
        self,
        action: str,
        details: Optional[Dict[str, Any]] = None,
        user: Optional[str] = None,
        ip_address: Optional[str] = None,
        success: bool = True,
    ) -> Optional[str]:
        """Log a mitigation-related event.

        Parameters
        ----------
        action : str
            Mitigation action name (e.g. ``"block_ip"``, ``"unblock_ip"``).
        details : dict, optional
            Additional structured info.
        user : str, optional
            Approver / triggering identity.
        ip_address : str, optional
            Target IP of the mitigation.
        success : bool
            Whether the mitigation succeeded.
        """
        return self.log(
            action=action,
            module="mitigation",
            details=details,
            user=user,
            ip_address=ip_address,
            success=success,
        )

    def log_detection(
        self,
        action: str,
        details: Optional[Dict[str, Any]] = None,
        user: Optional[str] = None,
        success: bool = True,
    ) -> Optional[str]:
        """Log a detection-related event."""
        return self.log(
            action=action,
            module="detection",
            details=details,
            user=user,
            success=success,
        )

    def log_system(
        self,
        action: str,
        details: Optional[Dict[str, Any]] = None,
        user: Optional[str] = None,
        success: bool = True,
    ) -> Optional[str]:
        """Log a general system event (startup, config change, etc.)."""
        return self.log(
            action=action,
            module="system",
            details=details,
            user=user,
            success=success,
        )

    def log_auth(
        self,
        action: str,
        details: Optional[Dict[str, Any]] = None,
        user: Optional[str] = None,
        ip_address: Optional[str] = None,
        success: bool = True,
    ) -> Optional[str]:
        """Log an authentication / authorisation event."""
        return self.log(
            action=action,
            module="auth",
            details=details,
            user=user,
            ip_address=ip_address,
            success=success,
        )

    # ------------------------------------------------------------------
    # Query helpers
    # ------------------------------------------------------------------

    def get_recent(self, limit: int = 50) -> List[dict]:
        """Return the *limit* most recent audit log entries, newest first.

        Returns
        -------
        list[dict]
            Each entry as a plain dict with all ``AuditLog`` fields.
        """
        try:
            session = SessionLocal()
            rows = (
                session.query(AuditLog)
                .order_by(AuditLog.timestamp.desc())
                .limit(limit)
                .all()
            )
            result = [self._row_to_dict(r) for r in rows]
            session.close()
            return result
        except Exception:
            logger.exception("Failed to query recent audit logs")
            return []

    def query_by_action(self, action: str, limit: int = 100) -> List[dict]:
        """Return audit entries matching *action*.

        Parameters
        ----------
        action : str
            The action string to filter on (exact match).
        limit : int
            Maximum entries to return.
        """
        try:
            session = SessionLocal()
            rows = (
                session.query(AuditLog)
                .filter(AuditLog.action == action)
                .order_by(AuditLog.timestamp.desc())
                .limit(limit)
                .all()
            )
            result = [self._row_to_dict(r) for r in rows]
            session.close()
            return result
        except Exception:
            logger.exception("Failed to query audit logs by action=%s", action)
            return []

    def query_by_module(self, module: str, limit: int = 100) -> List[dict]:
        """Return audit entries from *module*.

        Parameters
        ----------
        module : str
            Module name to filter on (e.g. ``"mitigation"``).
        limit : int
            Maximum entries to return.
        """
        try:
            session = SessionLocal()
            rows = (
                session.query(AuditLog)
                .filter(AuditLog.module == module)
                .order_by(AuditLog.timestamp.desc())
                .limit(limit)
                .all()
            )
            result = [self._row_to_dict(r) for r in rows]
            session.close()
            return result
        except Exception:
            logger.exception("Failed to query audit logs by module=%s", module)
            return []

    def query_by_user(self, user: str, limit: int = 100) -> List[dict]:
        """Return audit entries for *user*.

        Parameters
        ----------
        user : str
            User identifier.
        limit : int
            Maximum entries to return.
        """
        try:
            session = SessionLocal()
            rows = (
                session.query(AuditLog)
                .filter(AuditLog.user == user)
                .order_by(AuditLog.timestamp.desc())
                .limit(limit)
                .all()
            )
            result = [self._row_to_dict(r) for r in rows]
            session.close()
            return result
        except Exception:
            logger.exception("Failed to query audit logs by user=%s", user)
            return []

    def query_by_time_range(
        self,
        start: datetime,
        end: Optional[datetime] = None,
        limit: int = 500,
    ) -> List[dict]:
        """Return audit entries within a time window.

        Parameters
        ----------
        start : datetime
            Inclusive lower bound.
        end : datetime, optional
            Inclusive upper bound. Defaults to *now* (UTC).
        limit : int
            Maximum entries to return.
        """
        end = end or utcnow()
        try:
            session = SessionLocal()
            rows = (
                session.query(AuditLog)
                .filter(AuditLog.timestamp >= start, AuditLog.timestamp <= end)
                .order_by(AuditLog.timestamp.desc())
                .limit(limit)
                .all()
            )
            result = [self._row_to_dict(r) for r in rows]
            session.close()
            return result
        except Exception:
            logger.exception("Failed to query audit logs by time range")
            return []

    def query_failures(self, limit: int = 100) -> List[dict]:
        """Return audit entries where ``success`` is *False*."""
        try:
            session = SessionLocal()
            rows = (
                session.query(AuditLog)
                .filter(AuditLog.success == False)  # noqa: E712
                .order_by(AuditLog.timestamp.desc())
                .limit(limit)
                .all()
            )
            result = [self._row_to_dict(r) for r in rows]
            session.close()
            return result
        except Exception:
            logger.exception("Failed to query failure audit logs")
            return []

    # ------------------------------------------------------------------
    # Serialisation
    # ------------------------------------------------------------------

    @staticmethod
    def _row_to_dict(row: AuditLog) -> dict:
        """Convert an ``AuditLog`` ORM row to a plain dict."""
        return {
            "id": row.id,
            "timestamp": row.timestamp.isoformat() if row.timestamp else None,
            "action": row.action,
            "module": row.module,
            "details": row.details,
            "user": row.user,
            "ip_address": row.ip_address,
            "success": row.success,
            "created_at": row.created_at.isoformat() if row.created_at else None,
        }
