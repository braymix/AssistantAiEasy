"""
Step 0 — Pre-migration checks.

Validates that all prerequisites are met before starting the
mini → enterprise migration:

- Current profile and version
- Disk space availability
- PostgreSQL connectivity and version
- Qdrant connectivity
- Redis connectivity
- Existing backup verification
- Data integrity in source database

Usage::

    python scripts/migrate_enterprise/00_pre_check.py
    python scripts/migrate_enterprise/00_pre_check.py --json
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import shutil
import sqlite3
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger("migrate.pre_check")

# Minimum requirements
MIN_DISK_GB = 2.0
MIN_POSTGRES_VERSION = 14
REQUIRED_TABLES = [
    "conversations", "messages", "contexts",
    "knowledge_items", "detection_rules", "documents",
]


@dataclass
class CheckResult:
    """Result of a single pre-flight check."""
    name: str
    passed: bool
    message: str
    details: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "passed": self.passed,
            "message": self.message,
            "details": self.details,
        }


# ---------------------------------------------------------------------------
# Individual checks
# ---------------------------------------------------------------------------

def check_current_profile() -> CheckResult:
    """Verify current profile is mini (source for migration)."""
    profile = os.environ.get("KNOWLEDGEHUB_PROFILE", "mini")
    db_url = os.environ.get(
        "KNOWLEDGEHUB_DATABASE_URL",
        "sqlite+aiosqlite:///./data/sqlite/knowledgehub.db",
    )

    is_sqlite = "sqlite" in db_url
    return CheckResult(
        name="current_profile",
        passed=is_sqlite,
        message=f"Profile: {profile}, DB: {'SQLite' if is_sqlite else 'PostgreSQL'}",
        details={"profile": profile, "database_url": db_url[:50]},
    )


def check_disk_space(data_dir: str = ".") -> CheckResult:
    """Verify sufficient disk space for migration."""
    usage = shutil.disk_usage(data_dir)
    free_gb = usage.free / (1024 ** 3)
    total_gb = usage.total / (1024 ** 3)

    return CheckResult(
        name="disk_space",
        passed=free_gb >= MIN_DISK_GB,
        message=f"{free_gb:.1f} GB free of {total_gb:.1f} GB "
                f"(minimum {MIN_DISK_GB} GB required)",
        details={"free_gb": round(free_gb, 2), "total_gb": round(total_gb, 2)},
    )


def check_source_database() -> CheckResult:
    """Verify the source SQLite database exists and contains data."""
    db_url = os.environ.get(
        "KNOWLEDGEHUB_DATABASE_URL",
        "sqlite+aiosqlite:///./data/sqlite/knowledgehub.db",
    )
    db_path = db_url.split("///")[-1]

    if not Path(db_path).exists():
        return CheckResult(
            name="source_database",
            passed=False,
            message=f"SQLite database not found: {db_path}",
        )

    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()

        # Check tables exist
        cursor.execute(
            "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
        )
        tables = [row[0] for row in cursor.fetchall()]

        missing = [t for t in REQUIRED_TABLES if t not in tables]
        if missing:
            conn.close()
            return CheckResult(
                name="source_database",
                passed=False,
                message=f"Missing tables: {', '.join(missing)}",
                details={"found_tables": tables, "missing": missing},
            )

        # Count rows
        counts: dict[str, int] = {}
        for table in REQUIRED_TABLES:
            cursor.execute(f"SELECT COUNT(*) FROM [{table}]")  # noqa: S608
            counts[table] = cursor.fetchone()[0]

        total = sum(counts.values())
        conn.close()

        return CheckResult(
            name="source_database",
            passed=True,
            message=f"Database OK — {total} total rows across {len(tables)} tables",
            details={"row_counts": counts, "tables": tables},
        )
    except Exception as exc:
        return CheckResult(
            name="source_database",
            passed=False,
            message=f"Database error: {exc}",
        )


async def check_postgres() -> CheckResult:
    """Verify PostgreSQL is reachable and meets version requirements."""
    pg_url = os.environ.get(
        "KNOWLEDGEHUB_ENTERPRISE_POSTGRES_URL",
        os.environ.get(
            "KNOWLEDGEHUB_TARGET_DATABASE_URL",
            "postgresql+asyncpg://knowledgehub:changeme@postgres:5432/knowledgehub",
        ),
    )

    try:
        import asyncpg  # noqa: F401

        # Strip SQLAlchemy prefix
        dsn = pg_url.replace("postgresql+asyncpg://", "postgresql://")
        dsn = dsn.replace("postgresql+psycopg2://", "postgresql://")

        conn = await asyncpg.connect(dsn, timeout=10)
        version_str = await conn.fetchval("SELECT version()")
        server_version = conn.get_server_version()
        await conn.close()

        major = server_version.major
        passed = major >= MIN_POSTGRES_VERSION

        return CheckResult(
            name="postgresql",
            passed=passed,
            message=f"PostgreSQL {major} ({'OK' if passed else f'< {MIN_POSTGRES_VERSION}'})",
            details={"version": version_str, "major": major},
        )
    except ImportError:
        return CheckResult(
            name="postgresql",
            passed=False,
            message="asyncpg not installed — run: pip install asyncpg",
        )
    except Exception as exc:
        return CheckResult(
            name="postgresql",
            passed=False,
            message=f"Cannot connect to PostgreSQL: {exc}",
        )


async def check_qdrant() -> CheckResult:
    """Verify Qdrant is reachable."""
    host = os.environ.get("KNOWLEDGEHUB_QDRANT_HOST", "qdrant")
    port = int(os.environ.get("KNOWLEDGEHUB_QDRANT_PORT", "6333"))
    url = f"http://{host}:{port}"

    try:
        import httpx

        async with httpx.AsyncClient(timeout=10) as client:
            resp = await client.get(f"{url}/collections")
            data = resp.json()

        return CheckResult(
            name="qdrant",
            passed=resp.status_code == 200,
            message=f"Qdrant reachable at {url}",
            details={"collections": len(data.get("result", {}).get("collections", []))},
        )
    except ImportError:
        return CheckResult(
            name="qdrant",
            passed=False,
            message="httpx not installed — run: pip install httpx",
        )
    except Exception as exc:
        return CheckResult(
            name="qdrant",
            passed=False,
            message=f"Cannot connect to Qdrant at {url}: {exc}",
        )


async def check_redis() -> CheckResult:
    """Verify Redis is reachable (optional — only needed for clustering)."""
    redis_url = os.environ.get("KNOWLEDGEHUB_ENTERPRISE_REDIS_URL", "")

    if not redis_url:
        return CheckResult(
            name="redis",
            passed=True,
            message="Redis not configured (optional for clustering)",
            details={"required": False},
        )

    try:
        import redis.asyncio as aioredis

        client = aioredis.from_url(redis_url, socket_connect_timeout=5)
        info = await client.info("server")
        version = info.get("redis_version", "unknown")
        await client.close()

        return CheckResult(
            name="redis",
            passed=True,
            message=f"Redis {version} reachable",
            details={"version": version},
        )
    except ImportError:
        return CheckResult(
            name="redis",
            passed=True,
            message="redis package not installed (optional)",
            details={"required": False},
        )
    except Exception as exc:
        return CheckResult(
            name="redis",
            passed=False,
            message=f"Cannot connect to Redis: {exc}",
        )


def check_backup_exists(backup_dir: str = "./data/backups") -> CheckResult:
    """Check if a recent backup exists."""
    backup_path = Path(backup_dir)
    if not backup_path.exists():
        return CheckResult(
            name="backup",
            passed=False,
            message=f"Backup directory does not exist: {backup_dir}. "
                    "Run step 01_backup_current.py first.",
        )

    backups = sorted(backup_path.glob("*.gz"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not backups:
        return CheckResult(
            name="backup",
            passed=False,
            message="No backup files found. Run step 01_backup_current.py first.",
        )

    latest = backups[0]
    age_hours = (Path(latest).stat().st_mtime - __import__("time").time()) / -3600

    return CheckResult(
        name="backup",
        passed=age_hours < 24,
        message=f"Latest backup: {latest.name} ({age_hours:.1f}h ago)",
        details={
            "latest": str(latest),
            "age_hours": round(age_hours, 1),
            "count": len(backups),
        },
    )


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

async def run_all_checks(json_output: bool = False) -> bool:
    """Run all pre-migration checks and report results.

    Returns ``True`` if all critical checks pass.
    """
    logger.info("=" * 60)
    logger.info("KnowledgeHub — Pre-Migration Checks")
    logger.info("=" * 60)

    # Synchronous checks
    results: list[CheckResult] = [
        check_current_profile(),
        check_disk_space(),
        check_source_database(),
    ]

    # Async checks
    pg_result, qdrant_result, redis_result = await asyncio.gather(
        check_postgres(),
        check_qdrant(),
        check_redis(),
        return_exceptions=False,
    )
    results.extend([pg_result, qdrant_result, redis_result])

    # Backup check (informational — not blocking)
    results.append(check_backup_exists())

    # Report
    all_passed = True
    critical_failed: list[str] = []

    for r in results:
        status = "PASS" if r.passed else "FAIL"
        icon = "+" if r.passed else "!"
        logger.info("  [%s] %-20s %s", icon, r.name, r.message)
        if not r.passed and r.name not in ("backup", "redis"):
            all_passed = False
            critical_failed.append(r.name)

    logger.info("-" * 60)
    if all_passed:
        logger.info("All critical checks passed. Ready to migrate.")
    else:
        logger.error(
            "Critical checks FAILED: %s", ", ".join(critical_failed)
        )
        logger.error("Fix the issues above before proceeding.")

    if json_output:
        print(json.dumps({
            "passed": all_passed,
            "checks": [r.to_dict() for r in results],
        }, indent=2))

    return all_passed


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    json_output = "--json" in sys.argv
    passed = asyncio.run(run_all_checks(json_output=json_output))
    sys.exit(0 if passed else 1)


if __name__ == "__main__":
    main()
