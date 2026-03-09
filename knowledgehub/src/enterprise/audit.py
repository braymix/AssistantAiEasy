"""
Enterprise audit logging.

Provides an immutable audit trail of administrative actions for
compliance and forensics.  When ``audit_enabled`` is ``False`` (the
default), all write operations are silently skipped.

Features
────────
- Immutable append-only log (no updates / deletes on audit records)
- Structured entries with actor, action, resource, and diff
- Async-safe database writes
- Export to JSON / CSV for compliance reporting
- Configurable retention with automatic purging

Audit entry structure
─────────────────────
::

    {
        "id":          "uuid",
        "timestamp":   "2024-01-15T10:30:00Z",
        "actor_id":    "user-123",
        "actor_name":  "admin@company.com",
        "action":      "knowledge.verify",
        "resource_type": "knowledge_item",
        "resource_id": "ki-456",
        "tenant_id":   "default",
        "details":     {"verified": true, "verified_by": "admin"},
        "ip_address":  "192.168.1.100",
        "user_agent":  "Mozilla/5.0 ..."
    }
"""

from __future__ import annotations

import csv
import io
import json
import logging
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any

from src.enterprise import is_enterprise_enabled

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Audit entry
# ---------------------------------------------------------------------------

@dataclass(slots=True)
class AuditEntry:
    """A single audit log entry."""

    actor_id: str
    actor_name: str
    action: str
    resource_type: str
    resource_id: str = ""
    tenant_id: str = "default"
    details: dict[str, Any] = field(default_factory=dict)
    ip_address: str = ""
    user_agent: str = ""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["timestamp"] = self.timestamp.isoformat()
        return d


# ---------------------------------------------------------------------------
# Standard actions (constants to avoid typos)
# ---------------------------------------------------------------------------

class AuditAction:
    """Well-known audit action names."""

    # Rules
    RULE_CREATE = "rule.create"
    RULE_UPDATE = "rule.update"
    RULE_DELETE = "rule.delete"
    RULE_RELOAD = "rule.reload"

    # Contexts
    CONTEXT_CREATE = "context.create"
    CONTEXT_UPDATE = "context.update"
    CONTEXT_DELETE = "context.delete"

    # Knowledge
    KNOWLEDGE_CREATE = "knowledge.create"
    KNOWLEDGE_VERIFY = "knowledge.verify"
    KNOWLEDGE_REJECT = "knowledge.reject"
    KNOWLEDGE_DELETE = "knowledge.delete"
    KNOWLEDGE_IMPORT = "knowledge.import"
    KNOWLEDGE_EXPORT = "knowledge.export"

    # Auth
    AUTH_LOGIN = "auth.login"
    AUTH_LOGOUT = "auth.logout"
    AUTH_FAILED = "auth.failed"
    API_KEY_CREATE = "auth.apikey_create"
    API_KEY_REVOKE = "auth.apikey_revoke"

    # System
    BACKUP_CREATE = "system.backup_create"
    BACKUP_RESTORE = "system.backup_restore"
    CONFIG_CHANGE = "system.config_change"


# ---------------------------------------------------------------------------
# Audit logger
# ---------------------------------------------------------------------------

class AuditLogger:
    """Append-only audit log with pluggable backends.

    Default backend is in-memory (suitable for testing and mini mode).
    For production, subclass and override ``_persist`` to write to
    database or external log aggregator.
    """

    def __init__(self, max_memory_entries: int = 10_000) -> None:
        self._entries: list[AuditEntry] = []
        self._max_memory = max_memory_entries

    async def log(
        self,
        *,
        actor_id: str,
        actor_name: str,
        action: str,
        resource_type: str,
        resource_id: str = "",
        tenant_id: str = "default",
        details: dict[str, Any] | None = None,
        ip_address: str = "",
        user_agent: str = "",
    ) -> AuditEntry | None:
        """Record an audit entry.

        Returns the entry if audit is enabled, ``None`` otherwise.
        """
        if not is_enterprise_enabled("audit"):
            return None

        entry = AuditEntry(
            actor_id=actor_id,
            actor_name=actor_name,
            action=action,
            resource_type=resource_type,
            resource_id=resource_id,
            tenant_id=tenant_id,
            details=details or {},
            ip_address=ip_address,
            user_agent=user_agent,
        )

        await self._persist(entry)
        logger.info(
            "AUDIT: %s by %s on %s/%s",
            action, actor_name, resource_type, resource_id,
        )
        return entry

    async def _persist(self, entry: AuditEntry) -> None:
        """Store an entry.  Override for database / external backends."""
        self._entries.append(entry)
        # Evict oldest when over capacity
        if len(self._entries) > self._max_memory:
            self._entries = self._entries[-self._max_memory:]

    # -- Query interface ----------------------------------------------------

    async def query(
        self,
        *,
        actor_id: str | None = None,
        action: str | None = None,
        resource_type: str | None = None,
        resource_id: str | None = None,
        tenant_id: str | None = None,
        since: datetime | None = None,
        until: datetime | None = None,
        limit: int = 100,
        offset: int = 0,
    ) -> list[AuditEntry]:
        """Filter and paginate audit entries."""
        results = self._entries

        if actor_id:
            results = [e for e in results if e.actor_id == actor_id]
        if action:
            results = [e for e in results if e.action == action]
        if resource_type:
            results = [e for e in results if e.resource_type == resource_type]
        if resource_id:
            results = [e for e in results if e.resource_id == resource_id]
        if tenant_id:
            results = [e for e in results if e.tenant_id == tenant_id]
        if since:
            results = [e for e in results if e.timestamp >= since]
        if until:
            results = [e for e in results if e.timestamp <= until]

        # Sort newest first
        results = sorted(results, key=lambda e: e.timestamp, reverse=True)
        return results[offset : offset + limit]

    async def count(
        self,
        *,
        tenant_id: str | None = None,
        since: datetime | None = None,
    ) -> int:
        """Count entries matching the given filters."""
        entries = self._entries
        if tenant_id:
            entries = [e for e in entries if e.tenant_id == tenant_id]
        if since:
            entries = [e for e in entries if e.timestamp >= since]
        return len(entries)

    # -- Export -------------------------------------------------------------

    async def export_json(
        self,
        *,
        tenant_id: str | None = None,
        since: datetime | None = None,
        until: datetime | None = None,
    ) -> str:
        """Export matching entries as a JSON array string."""
        entries = await self.query(
            tenant_id=tenant_id, since=since, until=until, limit=999_999,
        )
        return json.dumps([e.to_dict() for e in entries], indent=2)

    async def export_csv(
        self,
        *,
        tenant_id: str | None = None,
        since: datetime | None = None,
        until: datetime | None = None,
    ) -> str:
        """Export matching entries as CSV."""
        entries = await self.query(
            tenant_id=tenant_id, since=since, until=until, limit=999_999,
        )
        if not entries:
            return ""

        buf = io.StringIO()
        fieldnames = [
            "id", "timestamp", "actor_id", "actor_name", "action",
            "resource_type", "resource_id", "tenant_id", "ip_address",
            "details",
        ]
        writer = csv.DictWriter(buf, fieldnames=fieldnames)
        writer.writeheader()
        for entry in entries:
            row = entry.to_dict()
            row["details"] = json.dumps(row.get("details", {}))
            row.pop("user_agent", None)
            writer.writerow(row)
        return buf.getvalue()

    # -- Retention ----------------------------------------------------------

    async def apply_retention(self, retention_days: int = 90) -> int:
        """Delete entries older than ``retention_days``.

        Returns the number of purged entries.
        """
        cutoff = datetime.now(timezone.utc) - timedelta(days=retention_days)
        before = len(self._entries)
        self._entries = [e for e in self._entries if e.timestamp >= cutoff]
        purged = before - len(self._entries)
        if purged:
            logger.info("Audit retention: purged %d entries older than %d days", purged, retention_days)
        return purged


# ---------------------------------------------------------------------------
# Singleton
# ---------------------------------------------------------------------------

_audit_logger = AuditLogger()


def get_audit_logger() -> AuditLogger:
    """Return the global audit logger instance."""
    return _audit_logger
