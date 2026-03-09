"""
Step 2 — Migrate SQLite → PostgreSQL.

Performs a full data migration from the mini-profile SQLite database
to the enterprise PostgreSQL instance:

1. Create the schema in PostgreSQL (via SQLAlchemy ``create_all``)
2. Read all rows from SQLite
3. Transform data types (JSON text → JSONB, datetime strings → timestamps)
4. Batch-insert into PostgreSQL
5. Verify row counts match
6. Create optimised indexes

Usage::

    python scripts/migrate_enterprise/02_migrate_database.py
    python scripts/migrate_enterprise/02_migrate_database.py --dry-run
    python scripts/migrate_enterprise/02_migrate_database.py --batch-size 500
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import sqlite3
import sys
import time
from datetime import datetime, timezone
from typing import Any

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger("migrate.database")

# Table migration order (respects foreign keys)
TABLE_ORDER = [
    "contexts",
    "conversations",
    "messages",
    "knowledge_items",
    "detection_rules",
    "documents",
]

# Columns per table (must match ORM models)
TABLE_COLUMNS: dict[str, list[str]] = {
    "contexts": [
        "id", "name", "description", "parent_id", "metadata_json", "created_at",
    ],
    "conversations": [
        "id", "session_id", "created_at", "updated_at", "metadata_json",
    ],
    "messages": [
        "id", "conversation_id", "role", "content", "created_at",
        "detected_contexts", "extracted_knowledge",
    ],
    "knowledge_items": [
        "id", "source_message_id", "content", "content_type", "contexts",
        "embedding_id", "verified", "created_at", "created_by",
    ],
    "detection_rules": [
        "id", "name", "description", "rule_type", "rule_config",
        "target_contexts", "priority", "enabled", "created_at", "updated_at",
    ],
    "documents": [
        "id", "title", "content", "metadata_json", "chunk_count",
        "created_at", "updated_at",
    ],
}

# Columns that contain JSON data (stored as TEXT in SQLite, JSONB in PG)
JSON_COLUMNS = {
    "metadata_json", "detected_contexts", "contexts",
    "rule_config", "target_contexts",
}

# Columns that contain booleans (stored as INTEGER in SQLite)
BOOL_COLUMNS = {"verified", "enabled", "extracted_knowledge"}

# Columns that contain timestamps
DATETIME_COLUMNS = {"created_at", "updated_at"}


# ---------------------------------------------------------------------------
# Data transformations
# ---------------------------------------------------------------------------

def _transform_row(
    table: str, columns: list[str], row: tuple,
) -> dict[str, Any]:
    """Transform a SQLite row into PostgreSQL-compatible values."""
    record: dict[str, Any] = {}
    for col, val in zip(columns, row):
        if val is None:
            record[col] = None
        elif col in JSON_COLUMNS:
            if isinstance(val, str):
                try:
                    record[col] = json.loads(val)
                except json.JSONDecodeError:
                    record[col] = val
            else:
                record[col] = val
        elif col in BOOL_COLUMNS:
            record[col] = bool(val)
        elif col in DATETIME_COLUMNS:
            if isinstance(val, str):
                # Parse ISO format from SQLite
                try:
                    record[col] = datetime.fromisoformat(val.replace("Z", "+00:00"))
                except ValueError:
                    record[col] = val
            else:
                record[col] = val
        else:
            record[col] = val
    return record


def _read_sqlite_table(
    conn: sqlite3.Connection, table: str, columns: list[str],
) -> list[dict[str, Any]]:
    """Read all rows from a SQLite table."""
    col_list = ", ".join(f"[{c}]" for c in columns)
    cursor = conn.execute(f"SELECT {col_list} FROM [{table}]")  # noqa: S608
    rows = cursor.fetchall()
    return [_transform_row(table, columns, row) for row in rows]


# ---------------------------------------------------------------------------
# PostgreSQL operations
# ---------------------------------------------------------------------------

async def _create_pg_schema(pg_url: str) -> None:
    """Create all tables in PostgreSQL using SQLAlchemy models."""
    from sqlalchemy.ext.asyncio import create_async_engine

    # Import all models so they register with Base
    import src.shared.models  # noqa: F401
    from src.shared.models import Base

    engine = create_async_engine(pg_url, echo=False)
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    await engine.dispose()

    logger.info("PostgreSQL schema created")


async def _insert_batch(
    pg_url: str,
    table: str,
    columns: list[str],
    rows: list[dict[str, Any]],
    batch_size: int = 250,
) -> int:
    """Batch-insert rows into a PostgreSQL table."""
    if not rows:
        return 0

    import asyncpg

    dsn = pg_url.replace("postgresql+asyncpg://", "postgresql://")
    conn = await asyncpg.connect(dsn, timeout=30)

    total = 0
    try:
        for i in range(0, len(rows), batch_size):
            batch = rows[i : i + batch_size]

            # Build parameterised INSERT
            col_list = ", ".join(f'"{c}"' for c in columns)
            placeholders = ", ".join(
                "(" + ", ".join(f"${j}" for j in range(k * len(columns) + 1, (k + 1) * len(columns) + 1)) + ")"
                for k in range(len(batch))
            )
            sql = f'INSERT INTO "{table}" ({col_list}) VALUES {placeholders} ON CONFLICT DO NOTHING'

            # Flatten values, serialising dicts/lists to JSON strings for JSONB
            values: list[Any] = []
            for row in batch:
                for col in columns:
                    val = row.get(col)
                    if col in JSON_COLUMNS and isinstance(val, (dict, list)):
                        values.append(json.dumps(val))
                    else:
                        values.append(val)

            await conn.execute(sql, *values)
            total += len(batch)
            logger.debug("  %s: inserted batch %d–%d", table, i, i + len(batch))
    finally:
        await conn.close()

    return total


async def _verify_counts(
    pg_url: str, expected: dict[str, int],
) -> dict[str, dict[str, int]]:
    """Compare row counts between expected (SQLite) and actual (PG)."""
    import asyncpg

    dsn = pg_url.replace("postgresql+asyncpg://", "postgresql://")
    conn = await asyncpg.connect(dsn, timeout=10)

    results: dict[str, dict[str, int]] = {}
    for table, expected_count in expected.items():
        actual = await conn.fetchval(f'SELECT COUNT(*) FROM "{table}"')
        results[table] = {
            "expected": expected_count,
            "actual": actual,
            "match": actual == expected_count,
        }

    await conn.close()
    return results


async def _create_indexes(pg_url: str) -> None:
    """Create optimised indexes for enterprise workload."""
    import asyncpg

    dsn = pg_url.replace("postgresql+asyncpg://", "postgresql://")
    conn = await asyncpg.connect(dsn, timeout=30)

    indexes = [
        # Conversations
        'CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_conv_session '
        'ON conversations (session_id)',
        'CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_conv_created '
        'ON conversations (created_at DESC)',

        # Messages
        'CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_msg_conv_created '
        'ON messages (conversation_id, created_at)',
        'CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_msg_role '
        'ON messages (role)',

        # Knowledge items
        'CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_ki_verified '
        'ON knowledge_items (verified)',
        'CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_ki_content_type '
        'ON knowledge_items (content_type)',
        'CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_ki_embedding '
        'ON knowledge_items (embedding_id)',
        'CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_ki_created '
        'ON knowledge_items (created_at DESC)',

        # Detection rules
        'CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_rules_priority '
        'ON detection_rules (priority)',
        'CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_rules_enabled '
        'ON detection_rules (enabled)',
        'CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_rules_type '
        'ON detection_rules (rule_type)',

        # Documents
        'CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_docs_title '
        'ON documents (title)',
        'CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_docs_created '
        'ON documents (created_at DESC)',

        # Contexts
        'CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_ctx_parent '
        'ON contexts (parent_id)',

        # Full-text search on knowledge content (PostgreSQL-specific)
        "CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_ki_content_fts "
        "ON knowledge_items USING gin (to_tsvector('italian', content))",
    ]

    for idx_sql in indexes:
        try:
            await conn.execute(idx_sql)
            idx_name = idx_sql.split("IF NOT EXISTS ")[-1].split(" ON")[0]
            logger.info("  Index created: %s", idx_name)
        except Exception as exc:
            logger.warning("  Index creation warning: %s", exc)

    await conn.close()
    logger.info("Optimised indexes created")


# ---------------------------------------------------------------------------
# Main migration
# ---------------------------------------------------------------------------

async def migrate_database(
    dry_run: bool = False,
    batch_size: int = 250,
) -> dict[str, Any]:
    """Execute the SQLite → PostgreSQL migration.

    Parameters
    ----------
    dry_run:
        If ``True``, read and transform data but do not write to PG.
    batch_size:
        Number of rows to insert per batch.

    Returns
    -------
    dict
        Migration summary with row counts and verification status.
    """
    logger.info("=" * 60)
    logger.info("KnowledgeHub — Database Migration (SQLite → PostgreSQL)")
    if dry_run:
        logger.info("MODE: DRY RUN — no data will be written")
    logger.info("=" * 60)

    # Resolve paths
    sqlite_url = os.environ.get(
        "KNOWLEDGEHUB_DATABASE_URL",
        "sqlite+aiosqlite:///./data/sqlite/knowledgehub.db",
    )
    pg_url = os.environ.get(
        "KNOWLEDGEHUB_TARGET_DATABASE_URL",
        "postgresql+asyncpg://knowledgehub:changeme@postgres:5432/knowledgehub",
    )
    sqlite_path = sqlite_url.split("///")[-1]

    # Read source data
    logger.info("Reading source SQLite database: %s", sqlite_path)
    conn = sqlite3.connect(sqlite_path)
    source_data: dict[str, list[dict[str, Any]]] = {}
    source_counts: dict[str, int] = {}

    for table in TABLE_ORDER:
        columns = TABLE_COLUMNS[table]
        rows = _read_sqlite_table(conn, table, columns)
        source_data[table] = rows
        source_counts[table] = len(rows)
        logger.info("  %-20s %6d rows", table, len(rows))

    conn.close()
    total_rows = sum(source_counts.values())
    logger.info("Total source rows: %d", total_rows)

    if dry_run:
        logger.info("Dry run complete — %d rows would be migrated", total_rows)
        return {"dry_run": True, "source_counts": source_counts}

    # Create schema
    logger.info("Creating PostgreSQL schema...")
    await _create_pg_schema(pg_url)

    # Insert data
    logger.info("Migrating data...")
    start = time.time()
    migrated_counts: dict[str, int] = {}

    for table in TABLE_ORDER:
        columns = TABLE_COLUMNS[table]
        rows = source_data[table]
        if not rows:
            migrated_counts[table] = 0
            continue

        count = await _insert_batch(pg_url, table, columns, rows, batch_size)
        migrated_counts[table] = count
        logger.info("  %-20s %6d rows migrated", table, count)

    elapsed = time.time() - start
    logger.info("Data migration completed in %.1fs", elapsed)

    # Verify
    logger.info("Verifying row counts...")
    verification = await _verify_counts(pg_url, source_counts)
    all_match = all(v["match"] for v in verification.values())

    for table, v in verification.items():
        status = "OK" if v["match"] else "MISMATCH"
        logger.info(
            "  %-20s expected=%d actual=%d [%s]",
            table, v["expected"], v["actual"], status,
        )

    # Create indexes
    if all_match:
        logger.info("Creating optimised indexes...")
        await _create_indexes(pg_url)

    logger.info("-" * 60)
    if all_match:
        logger.info("Database migration SUCCESSFUL (%d rows in %.1fs)", total_rows, elapsed)
    else:
        logger.error("Database migration completed with VERIFICATION ERRORS")

    return {
        "source_counts": source_counts,
        "migrated_counts": migrated_counts,
        "verification": {k: dict(v) for k, v in verification.items()},
        "all_match": all_match,
        "elapsed_seconds": round(elapsed, 2),
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    dry_run = "--dry-run" in sys.argv
    batch_size = 250
    for i, arg in enumerate(sys.argv):
        if arg == "--batch-size" and i + 1 < len(sys.argv):
            batch_size = int(sys.argv[i + 1])

    result = asyncio.run(migrate_database(dry_run=dry_run, batch_size=batch_size))
    if not result.get("all_match", True) and not result.get("dry_run"):
        sys.exit(1)


if __name__ == "__main__":
    main()
