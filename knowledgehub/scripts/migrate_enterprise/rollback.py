"""
Rollback — Restore from a migration backup.

Reverses the enterprise migration by restoring the mini-profile
state from a backup created by ``01_backup_current.py``:

1. Locate backup by ID (timestamp) or use the latest
2. Restore SQLite database from compressed backup
3. Restore ChromaDB data from archive
4. Restore configuration files
5. Verify restored system works

Usage::

    # Rollback to a specific backup
    python scripts/migrate_enterprise/rollback.py --backup-id 20240115_120000

    # Rollback to the latest backup
    python scripts/migrate_enterprise/rollback.py --latest

    # Dry run (show what would be restored)
    python scripts/migrate_enterprise/rollback.py --latest --dry-run
"""

from __future__ import annotations

import gzip
import json
import logging
import shutil
import sys
from pathlib import Path
from typing import Any

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger("migrate.rollback")


def find_backup(
    backup_id: str | None = None,
    backup_dir: str = "./data/backups",
    latest: bool = False,
) -> Path | None:
    """Find a migration backup directory by ID or pick the latest."""
    base = Path(backup_dir)
    if not base.exists():
        logger.error("Backup directory does not exist: %s", backup_dir)
        return None

    if backup_id:
        target = base / f"migration_{backup_id}"
        if target.exists():
            return target
        logger.error("Backup not found: %s", target)
        return None

    if latest:
        candidates = sorted(
            base.glob("migration_*/manifest.json"),
            key=lambda p: p.parent.name,
            reverse=True,
        )
        if candidates:
            return candidates[0].parent
        logger.error("No migration backups found in %s", backup_dir)
        return None

    # List available
    candidates = sorted(base.glob("migration_*/manifest.json"))
    if not candidates:
        logger.error("No migration backups found in %s", backup_dir)
        return None

    logger.info("Available backups:")
    for c in candidates:
        manifest = json.loads(c.read_text())
        logger.info(
            "  %s  (rows: %s, verified: %s)",
            c.parent.name,
            sum(manifest.get("row_counts", {}).values()),
            manifest.get("verified", "?"),
        )
    logger.info("Specify --backup-id <id> or --latest")
    return None


def restore_sqlite(manifest: dict[str, Any], dry_run: bool = False) -> bool:
    """Restore the SQLite database from gzip backup."""
    sqlite_info = manifest.get("files", {}).get("sqlite", {})
    backup_path = sqlite_info.get("path", "")
    if not backup_path:
        logger.error("No SQLite backup path in manifest")
        return False

    backup_file = Path(backup_path)
    if not backup_file.exists():
        logger.error("SQLite backup file not found: %s", backup_path)
        return False

    # Determine target path from current config or default
    import os
    db_url = os.environ.get(
        "KNOWLEDGEHUB_DATABASE_URL",
        "sqlite+aiosqlite:///./data/sqlite/knowledgehub.db",
    )

    if "sqlite" not in db_url:
        logger.info("Current DB is not SQLite — skipping SQLite restore")
        logger.info("To restore, set KNOWLEDGEHUB_DATABASE_URL to a SQLite path")
        return True

    db_path = Path(db_url.split("///")[-1])

    if dry_run:
        logger.info("DRY RUN: would restore %s → %s", backup_file, db_path)
        return True

    # Create parent directory
    db_path.parent.mkdir(parents=True, exist_ok=True)

    # Decompress
    with gzip.open(backup_file, "rb") as f_in, open(db_path, "wb") as f_out:
        shutil.copyfileobj(f_in, f_out)

    logger.info("SQLite restored: %s → %s", backup_file.name, db_path)
    return True


def restore_chromadb(manifest: dict[str, Any], dry_run: bool = False) -> bool:
    """Restore ChromaDB data from tar.gz archive."""
    chroma_info = manifest.get("files", {}).get("chromadb", {})
    if chroma_info.get("skipped"):
        logger.info("ChromaDB was not backed up — skipping restore")
        return True

    backup_path = chroma_info.get("path", "")
    if not backup_path:
        logger.info("No ChromaDB backup in manifest — skipping")
        return True

    backup_file = Path(backup_path)
    if not backup_file.exists():
        logger.error("ChromaDB backup not found: %s", backup_path)
        return False

    import os
    chroma_dir = Path(os.environ.get("KNOWLEDGEHUB_CHROMA_PERSIST_DIR", "./data/chroma"))

    if dry_run:
        logger.info("DRY RUN: would restore %s → %s", backup_file, chroma_dir)
        return True

    # Remove existing and extract
    if chroma_dir.exists():
        shutil.rmtree(chroma_dir)
    chroma_dir.mkdir(parents=True, exist_ok=True)

    shutil.unpack_archive(str(backup_file), str(chroma_dir))
    logger.info("ChromaDB restored: %s → %s", backup_file.name, chroma_dir)
    return True


def restore_config(manifest: dict[str, Any], dry_run: bool = False) -> bool:
    """Restore configuration files from backup."""
    config_info = manifest.get("files", {}).get("config", {})
    config_dir = config_info.get("path", "")
    copied = config_info.get("copied", [])

    if not config_dir or not copied:
        logger.info("No config files in backup — skipping")
        return True

    config_path = Path(config_dir)
    if not config_path.exists():
        logger.error("Config backup directory not found: %s", config_dir)
        return False

    for name in copied:
        src = config_path / name
        dest = Path(name)

        if not src.exists():
            logger.warning("Config backup missing: %s", name)
            continue

        if dry_run:
            logger.info("DRY RUN: would restore %s", name)
            continue

        # Backup current before overwriting
        if dest.exists():
            dest.rename(f"{name}.pre_rollback")

        shutil.copy2(src, dest)
        logger.info("Config restored: %s", name)

    return True


def verify_rollback() -> bool:
    """Quick verification that the restored system is functional."""
    import os
    import sqlite3

    db_url = os.environ.get(
        "KNOWLEDGEHUB_DATABASE_URL",
        "sqlite+aiosqlite:///./data/sqlite/knowledgehub.db",
    )

    if "sqlite" not in db_url:
        logger.info("Non-SQLite DB — skipping rollback verification")
        return True

    db_path = db_url.split("///")[-1]
    if not Path(db_path).exists():
        logger.error("Restored database not found: %s", db_path)
        return False

    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
        )
        tables = [row[0] for row in cursor.fetchall()]
        conn.close()

        if tables:
            logger.info("Verified: %d tables in restored database", len(tables))
            return True
        else:
            logger.error("Restored database has no tables")
            return False
    except Exception as exc:
        logger.error("Verification failed: %s", exc)
        return False


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_rollback(
    backup_id: str | None = None,
    latest: bool = False,
    dry_run: bool = False,
) -> bool:
    """Execute a full rollback from a migration backup."""
    logger.info("=" * 60)
    logger.info("KnowledgeHub — Migration Rollback")
    if dry_run:
        logger.info("MODE: DRY RUN")
    logger.info("=" * 60)

    # Find backup
    backup_dir = find_backup(backup_id=backup_id, latest=latest)
    if backup_dir is None:
        return False

    manifest_path = backup_dir / "manifest.json"
    if not manifest_path.exists():
        logger.error("Manifest not found in backup: %s", backup_dir)
        return False

    manifest = json.loads(manifest_path.read_text())
    logger.info("Restoring from backup: %s", manifest.get("backup_id", "?"))
    logger.info("Backup timestamp: %s", manifest.get("timestamp", "?"))

    total_rows = sum(
        v for v in manifest.get("row_counts", {}).values() if isinstance(v, int) and v >= 0
    )
    logger.info("Expected data: %d rows", total_rows)

    # Execute rollback steps
    steps = [
        ("SQLite database", lambda: restore_sqlite(manifest, dry_run)),
        ("ChromaDB data", lambda: restore_chromadb(manifest, dry_run)),
        ("Configuration files", lambda: restore_config(manifest, dry_run)),
    ]

    success = True
    for name, step_fn in steps:
        logger.info("Restoring %s...", name)
        if not step_fn():
            logger.error("Failed to restore: %s", name)
            success = False

    # Verify
    if not dry_run and success:
        logger.info("Verifying restored system...")
        success = verify_rollback()

    logger.info("-" * 60)
    if dry_run:
        logger.info("Dry run complete — no changes made")
    elif success:
        logger.info("Rollback SUCCESSFUL")
        logger.info("Restart services: docker compose up -d")
    else:
        logger.error("Rollback completed with ERRORS")

    return success


def main() -> None:
    backup_id = None
    latest = "--latest" in sys.argv
    dry_run = "--dry-run" in sys.argv

    for i, arg in enumerate(sys.argv):
        if arg.startswith("--backup-id="):
            backup_id = arg.split("=", 1)[1]
        elif arg == "--backup-id" and i + 1 < len(sys.argv):
            backup_id = sys.argv[i + 1]

    if not backup_id and not latest:
        # Show available backups
        find_backup()
        sys.exit(1)

    success = run_rollback(backup_id=backup_id, latest=latest, dry_run=dry_run)
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
