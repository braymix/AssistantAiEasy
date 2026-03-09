"""
Step 1 — Full backup of the current mini deployment.

Creates a complete, verified backup of all data before migration:

- SQLite database (gzip compressed)
- ChromaDB data directory (tar.gz archive)
- Configuration files (.env, docker-compose overrides)
- Integrity verification (checksum + row counts)

The backup ID (timestamp) is used by subsequent steps and by the
rollback script to identify the restore point.

Usage::

    python scripts/migrate_enterprise/01_backup_current.py
    python scripts/migrate_enterprise/01_backup_current.py --output /mnt/backup
"""

from __future__ import annotations

import gzip
import hashlib
import json
import logging
import os
import shutil
import sqlite3
import sys
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger("migrate.backup")


@dataclass
class BackupManifest:
    """Describes a complete backup for verification and rollback."""

    backup_id: str
    timestamp: str
    backup_dir: str
    files: dict[str, dict[str, Any]] = field(default_factory=dict)
    row_counts: dict[str, int] = field(default_factory=dict)
    verified: bool = False

    def save(self, path: Path) -> None:
        path.write_text(json.dumps(asdict(self), indent=2))

    @classmethod
    def load(cls, path: Path) -> BackupManifest:
        data = json.loads(path.read_text())
        return cls(**data)


TABLES = [
    "conversations", "messages", "contexts",
    "knowledge_items", "detection_rules", "documents",
]


def _sha256(file_path: Path) -> str:
    """Compute SHA-256 of a file."""
    h = hashlib.sha256()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def _get_row_counts(db_path: str) -> dict[str, int]:
    """Get row counts for all tables."""
    conn = sqlite3.connect(db_path)
    counts: dict[str, int] = {}
    for table in TABLES:
        try:
            cursor = conn.execute(f"SELECT COUNT(*) FROM [{table}]")  # noqa: S608
            counts[table] = cursor.fetchone()[0]
        except sqlite3.OperationalError:
            counts[table] = -1
    conn.close()
    return counts


def backup_sqlite(db_path: str, backup_dir: Path) -> dict[str, Any]:
    """Compress the SQLite database with gzip."""
    src = Path(db_path)
    if not src.exists():
        raise FileNotFoundError(f"SQLite database not found: {db_path}")

    dest = backup_dir / f"{src.stem}.db.gz"
    original_size = src.stat().st_size

    with open(src, "rb") as f_in, gzip.open(dest, "wb", compresslevel=9) as f_out:
        shutil.copyfileobj(f_in, f_out)

    compressed_size = dest.stat().st_size
    ratio = (1 - compressed_size / original_size) * 100 if original_size else 0

    logger.info(
        "SQLite backup: %s → %s (%.1f%% compression)",
        src.name, dest.name, ratio,
    )

    return {
        "path": str(dest),
        "original_size": original_size,
        "compressed_size": compressed_size,
        "checksum": _sha256(dest),
    }


def backup_chromadb(chroma_dir: str, backup_dir: Path) -> dict[str, Any]:
    """Archive the ChromaDB persist directory."""
    src = Path(chroma_dir)
    if not src.exists():
        logger.warning("ChromaDB directory not found: %s — skipping", chroma_dir)
        return {"path": "", "skipped": True}

    archive_name = "chromadb_data"
    archive_path = Path(
        shutil.make_archive(str(backup_dir / archive_name), "gztar", str(src))
    )
    size = archive_path.stat().st_size

    logger.info("ChromaDB backup: %s (%.1f MB)", archive_path.name, size / (1024 * 1024))

    return {
        "path": str(archive_path),
        "size": size,
        "checksum": _sha256(archive_path),
    }


def backup_config_files(backup_dir: Path) -> dict[str, Any]:
    """Copy configuration files."""
    config_dir = backup_dir / "config"
    config_dir.mkdir(exist_ok=True)

    copied: list[str] = []
    config_files = [
        ".env",
        "docker-compose.yml",
        "docker-compose.override.yml",
        "docker-compose.enterprise.yml",
        "docker-compose.dev.yml",
    ]

    for name in config_files:
        src = Path(name)
        if src.exists():
            shutil.copy2(src, config_dir / name)
            copied.append(name)
            logger.info("Config backup: %s", name)

    return {"copied": copied, "path": str(config_dir)}


def verify_backup(manifest: BackupManifest, db_path: str) -> bool:
    """Verify backup integrity by checking checksums and row counts."""
    logger.info("Verifying backup integrity...")
    errors: list[str] = []

    # Verify checksums
    for name, info in manifest.files.items():
        file_path = info.get("path", "")
        expected = info.get("checksum", "")
        if not file_path or not expected or info.get("skipped"):
            continue

        p = Path(file_path)
        if not p.exists():
            errors.append(f"{name}: file missing at {file_path}")
            continue

        actual = _sha256(p)
        if actual != expected:
            errors.append(f"{name}: checksum mismatch")
        else:
            logger.info("  [+] %s checksum OK", name)

    # Verify row counts match source
    if Path(db_path).exists():
        source_counts = _get_row_counts(db_path)
        for table, expected_count in manifest.row_counts.items():
            actual_count = source_counts.get(table, -1)
            if actual_count != expected_count:
                errors.append(
                    f"Row count mismatch for {table}: "
                    f"expected {expected_count}, got {actual_count}"
                )
            else:
                logger.info("  [+] %s: %d rows OK", table, actual_count)

    if errors:
        for err in errors:
            logger.error("  [!] %s", err)
        return False

    logger.info("Backup verification PASSED")
    return True


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_backup(output_dir: str | None = None) -> BackupManifest:
    """Execute a full backup and return the manifest.

    Parameters
    ----------
    output_dir:
        Override the default backup directory (``./data/backups``).

    Returns
    -------
    BackupManifest
        Metadata describing all backed-up files.
    """
    backup_id = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    base_dir = Path(output_dir or "./data/backups")
    backup_dir = base_dir / f"migration_{backup_id}"
    backup_dir.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 60)
    logger.info("KnowledgeHub — Full Backup (ID: %s)", backup_id)
    logger.info("Output: %s", backup_dir)
    logger.info("=" * 60)

    # Source paths from environment
    db_url = os.environ.get(
        "KNOWLEDGEHUB_DATABASE_URL",
        "sqlite+aiosqlite:///./data/sqlite/knowledgehub.db",
    )
    db_path = db_url.split("///")[-1]
    chroma_dir = os.environ.get("KNOWLEDGEHUB_CHROMA_PERSIST_DIR", "./data/chroma")

    manifest = BackupManifest(
        backup_id=backup_id,
        timestamp=datetime.now(timezone.utc).isoformat(),
        backup_dir=str(backup_dir),
    )

    # 1. SQLite
    start = time.time()
    manifest.files["sqlite"] = backup_sqlite(db_path, backup_dir)
    manifest.row_counts = _get_row_counts(db_path)
    logger.info("SQLite backup completed in %.1fs", time.time() - start)

    # 2. ChromaDB
    start = time.time()
    manifest.files["chromadb"] = backup_chromadb(chroma_dir, backup_dir)
    logger.info("ChromaDB backup completed in %.1fs", time.time() - start)

    # 3. Config files
    manifest.files["config"] = backup_config_files(backup_dir)

    # 4. Verify
    manifest.verified = verify_backup(manifest, db_path)

    # 5. Save manifest
    manifest_path = backup_dir / "manifest.json"
    manifest.save(manifest_path)
    logger.info("Manifest saved: %s", manifest_path)

    logger.info("-" * 60)
    if manifest.verified:
        logger.info("Backup COMPLETE and VERIFIED: %s", backup_id)
    else:
        logger.error("Backup completed but verification FAILED")

    total_rows = sum(v for v in manifest.row_counts.values() if v >= 0)
    logger.info("Total rows backed up: %d", total_rows)

    return manifest


def main() -> None:
    output = None
    for i, arg in enumerate(sys.argv[1:], 1):
        if arg == "--output" and i < len(sys.argv) - 1:
            output = sys.argv[i + 1]

    manifest = run_backup(output_dir=output)
    if not manifest.verified:
        sys.exit(1)


if __name__ == "__main__":
    main()
