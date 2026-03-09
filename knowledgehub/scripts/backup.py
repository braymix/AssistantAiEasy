"""Backup utility for KnowledgeHub.

Creates compressed backups of:
  - SQLite database (or PostgreSQL dump)
  - ChromaDB data directory
  - Configuration files

Supports retention policy to auto-delete old backups.

Usage:
    python scripts/backup.py                     # full backup
    python scripts/backup.py --db-only           # database only
    python scripts/backup.py --retention-days 30 # keep last 30 days
    python scripts/backup.py --output /tmp        # custom output directory
"""

import asyncio
import gzip
import shutil
import sys
from datetime import datetime, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import get_settings
from src.config.logging import setup_logging, get_logger

logger = get_logger(__name__)

# Default retention: 7 days
DEFAULT_RETENTION_DAYS = 7
BACKUP_PREFIX = "knowledgehub_backup"


def _parse_args() -> dict:
    """Parse CLI arguments into a dict."""
    args = {
        "db_only": False,
        "retention_days": DEFAULT_RETENTION_DAYS,
        "output_dir": None,
    }
    argv = sys.argv[1:]
    i = 0
    while i < len(argv):
        if argv[i] == "--db-only":
            args["db_only"] = True
        elif argv[i] == "--retention-days" and i + 1 < len(argv):
            i += 1
            args["retention_days"] = int(argv[i])
        elif argv[i] == "--output" and i + 1 < len(argv):
            i += 1
            args["output_dir"] = argv[i]
        i += 1
    return args


def _get_backup_dir(output_dir: str | None, settings) -> Path:
    """Determine backup directory."""
    if output_dir:
        backup_dir = Path(output_dir)
    elif settings.is_sqlite:
        db_path = Path(settings.database_url.split("///")[-1])
        backup_dir = db_path.parent.parent / "backups"
    else:
        backup_dir = Path("data/backups")
    backup_dir.mkdir(parents=True, exist_ok=True)
    return backup_dir


def _backup_sqlite(settings, backup_dir: Path, timestamp: str) -> Path | None:
    """Backup SQLite database file with gzip compression."""
    db_path = Path(settings.database_url.split("///")[-1])
    if not db_path.exists():
        logger.warning("sqlite_db_not_found", path=str(db_path))
        return None

    backup_file = backup_dir / f"{BACKUP_PREFIX}_db_{timestamp}.sqlite.gz"

    with open(db_path, "rb") as f_in:
        with gzip.open(backup_file, "wb", compresslevel=6) as f_out:
            shutil.copyfileobj(f_in, f_out)

    original_size = db_path.stat().st_size
    compressed_size = backup_file.stat().st_size
    ratio = (1 - compressed_size / original_size) * 100 if original_size > 0 else 0

    logger.info(
        "sqlite_backup_created",
        path=str(backup_file),
        original_bytes=original_size,
        compressed_bytes=compressed_size,
        compression_ratio=f"{ratio:.1f}%",
    )
    return backup_file


async def _backup_postgres(settings, backup_dir: Path, timestamp: str) -> Path | None:
    """Backup PostgreSQL database via pg_dump."""
    backup_file = backup_dir / f"{BACKUP_PREFIX}_db_{timestamp}.sql.gz"

    # Extract connection info from URL
    # Format: postgresql+asyncpg://user:pass@host:port/dbname
    url = settings.database_url
    # Remove driver prefix
    clean_url = url.replace("postgresql+asyncpg://", "").replace("postgresql://", "")
    userpass, hostdb = clean_url.split("@", 1)
    user, password = userpass.split(":", 1) if ":" in userpass else (userpass, "")
    hostport, dbname = hostdb.split("/", 1)
    host, port = hostport.split(":", 1) if ":" in hostport else (hostport, "5432")

    cmd = (
        f"PGPASSWORD={password} pg_dump -h {host} -p {port} -U {user} -d {dbname} "
        f"--no-owner --no-acl | gzip > {backup_file}"
    )

    proc = await asyncio.create_subprocess_shell(
        cmd,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    _, stderr = await proc.communicate()

    if proc.returncode != 0:
        logger.error("pg_dump_failed", stderr=stderr.decode())
        return None

    logger.info(
        "postgres_backup_created",
        path=str(backup_file),
        size_bytes=backup_file.stat().st_size,
    )
    return backup_file


def _backup_chroma(settings, backup_dir: Path, timestamp: str) -> Path | None:
    """Backup ChromaDB data directory."""
    chroma_dir = Path(settings.chroma_persist_dir)
    if not chroma_dir.exists():
        logger.info("chroma_dir_not_found_skipping", path=str(chroma_dir))
        return None

    archive_name = f"{BACKUP_PREFIX}_chroma_{timestamp}"
    archive_path = backup_dir / archive_name

    shutil.make_archive(str(archive_path), "gztar", chroma_dir.parent, chroma_dir.name)

    result_path = Path(f"{archive_path}.tar.gz")
    logger.info(
        "chroma_backup_created",
        path=str(result_path),
        size_bytes=result_path.stat().st_size,
    )
    return result_path


def _apply_retention(backup_dir: Path, retention_days: int) -> int:
    """Delete backups older than retention_days. Returns count deleted."""
    if retention_days <= 0:
        return 0

    cutoff = datetime.now() - timedelta(days=retention_days)
    deleted = 0

    for f in backup_dir.iterdir():
        if f.name.startswith(BACKUP_PREFIX) and f.is_file():
            mtime = datetime.fromtimestamp(f.stat().st_mtime)
            if mtime < cutoff:
                f.unlink()
                deleted += 1
                logger.info("deleted_old_backup", file=f.name, age_days=(datetime.now() - mtime).days)

    if deleted:
        logger.info("retention_applied", deleted=deleted, retention_days=retention_days)
    return deleted


async def main() -> None:
    settings = get_settings()
    setup_logging(settings)

    args = _parse_args()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_dir = _get_backup_dir(args["output_dir"], settings)

    logger.info(
        "backup_start",
        profile=settings.profile.value,
        db_only=args["db_only"],
        output=str(backup_dir),
    )

    created_files: list[str] = []

    # ── Database backup ─────────────────────────────────────────────────────
    if settings.is_sqlite:
        result = _backup_sqlite(settings, backup_dir, timestamp)
    else:
        result = await _backup_postgres(settings, backup_dir, timestamp)

    if result:
        created_files.append(str(result))

    # ── Vector store backup (skip if --db-only) ─────────────────────────────
    if not args["db_only"] and settings.vectorstore_backend.value == "chroma":
        result = _backup_chroma(settings, backup_dir, timestamp)
        if result:
            created_files.append(str(result))

    # ── Retention policy ────────────────────────────────────────────────────
    _apply_retention(backup_dir, args["retention_days"])

    # ── Summary ─────────────────────────────────────────────────────────────
    if created_files:
        logger.info("backup_complete", files=created_files, count=len(created_files))
        print(f"\nBackup complete ({len(created_files)} file(s)):")
        for f in created_files:
            print(f"  {f}")
    else:
        logger.warning("no_backups_created")
        print("\nNo backups were created.")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
