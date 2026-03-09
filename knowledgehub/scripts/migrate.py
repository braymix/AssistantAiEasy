"""Database migration helper with backup and rollback support.

Simplified Alembic-style migration system:
  - Backs up the database before applying changes
  - Creates/updates tables via SQLAlchemy metadata
  - Rolls back from backup on failure

Usage:
    python scripts/migrate.py                # apply migrations (upgrade)
    python scripts/migrate.py upgrade        # same as above
    python scripts/migrate.py reset          # drop all + recreate (destructive)
    python scripts/migrate.py status         # show current tables
"""

import asyncio
import shutil
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import get_settings
from src.config.logging import setup_logging, get_logger
from src.shared.database import Base, dispose_engine

logger = get_logger(__name__)


def _backup_sqlite(db_url: str) -> Path | None:
    """Create a timestamped backup of the SQLite database.

    Returns the backup path, or None if the DB file doesn't exist yet.
    """
    db_path = Path(db_url.split("///")[-1])
    if not db_path.exists():
        logger.info("no_existing_db_to_backup", path=str(db_path))
        return None

    backup_dir = db_path.parent / "backups"
    backup_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_path = backup_dir / f"{db_path.stem}_{timestamp}{db_path.suffix}"

    shutil.copy2(db_path, backup_path)
    logger.info("backup_created", path=str(backup_path), size_bytes=backup_path.stat().st_size)
    return backup_path


def _restore_sqlite(db_url: str, backup_path: Path) -> None:
    """Restore SQLite database from backup."""
    db_path = Path(db_url.split("///")[-1])
    shutil.copy2(backup_path, db_path)
    logger.info("database_restored_from_backup", backup=str(backup_path))


async def _get_current_tables(engine) -> list[str]:
    """Return list of table names in the database."""
    from sqlalchemy import inspect as sa_inspect

    async with engine.connect() as conn:
        tables = await conn.run_sync(lambda sync_conn: sa_inspect(sync_conn).get_table_names())
    return sorted(tables)


async def action_upgrade(settings) -> None:
    """Apply schema changes (create missing tables/columns)."""
    from sqlalchemy.ext.asyncio import create_async_engine

    import src.shared.models  # noqa: F401 – register all models

    # Backup before migration (SQLite only)
    backup_path = None
    if settings.is_sqlite:
        backup_path = _backup_sqlite(settings.database_url)

    engine = create_async_engine(settings.database_url)

    try:
        # Get tables before migration
        tables_before = await _get_current_tables(engine)
        logger.info("tables_before_migration", tables=tables_before)

        # Apply create_all (idempotent – adds missing tables)
        async with engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)

        # Get tables after migration
        tables_after = await _get_current_tables(engine)
        new_tables = set(tables_after) - set(tables_before)

        if new_tables:
            logger.info("migration_applied", new_tables=sorted(new_tables))
        else:
            logger.info("no_schema_changes_needed", tables=tables_after)

    except Exception:
        logger.exception("migration_failed")
        # Rollback from backup if available
        if backup_path:
            _restore_sqlite(settings.database_url, backup_path)
            logger.info("rolled_back_from_backup")
        await engine.dispose()
        sys.exit(1)

    await engine.dispose()
    logger.info("migration_upgrade_complete")


async def action_reset(settings) -> None:
    """Drop all tables and recreate (destructive!)."""
    from sqlalchemy.ext.asyncio import create_async_engine

    import src.shared.models  # noqa: F401

    # Backup before reset (SQLite only)
    if settings.is_sqlite:
        _backup_sqlite(settings.database_url)

    engine = create_async_engine(settings.database_url)

    try:
        async with engine.begin() as conn:
            await conn.run_sync(Base.metadata.drop_all)
            logger.info("all_tables_dropped")
            await conn.run_sync(Base.metadata.create_all)
            logger.info("all_tables_recreated")
    except Exception:
        logger.exception("reset_failed")
        await engine.dispose()
        sys.exit(1)

    await engine.dispose()
    logger.info("database_reset_complete")


async def action_status(settings) -> None:
    """Show current database tables."""
    from sqlalchemy.ext.asyncio import create_async_engine

    engine = create_async_engine(settings.database_url)

    try:
        tables = await _get_current_tables(engine)
        logger.info("current_tables", tables=tables, count=len(tables))
        print(f"\nDatabase: {settings.database_url}")
        print(f"Tables ({len(tables)}):")
        for t in tables:
            print(f"  - {t}")
    except Exception:
        logger.exception("status_check_failed")
        await engine.dispose()
        sys.exit(1)

    await engine.dispose()


ACTIONS = {
    "upgrade": action_upgrade,
    "reset": action_reset,
    "status": action_status,
}


async def main() -> None:
    settings = get_settings()
    setup_logging(settings)

    action = sys.argv[1] if len(sys.argv) > 1 else "upgrade"

    if action not in ACTIONS:
        print(f"Unknown action: {action}")
        print(f"Usage: python scripts/migrate.py [{' | '.join(ACTIONS)}]")
        sys.exit(1)

    logger.info("migrate_start", action=action, profile=settings.profile.value)
    await ACTIONS[action](settings)
    await dispose_engine()


if __name__ == "__main__":
    asyncio.run(main())
