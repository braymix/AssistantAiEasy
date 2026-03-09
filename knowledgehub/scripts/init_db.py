"""Initialize the KnowledgeHub database.

Idempotent script that:
  1. Creates the data directory (SQLite) if needed
  2. Creates all tables via Base.metadata.create_all (safe to re-run)
  3. Optionally applies seed data (--seed flag)

Usage:
    python scripts/init_db.py          # create tables only
    python scripts/init_db.py --seed   # create tables + seed rules
"""

import asyncio
import sys
from pathlib import Path

# Ensure project root is on sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import get_settings
from src.config.logging import setup_logging, get_logger
from src.shared.database import init_db, dispose_engine

logger = get_logger(__name__)


async def main() -> None:
    settings = get_settings()
    setup_logging(settings)

    seed = "--seed" in sys.argv

    logger.info(
        "initializing_database",
        profile=settings.profile.value,
        database=settings.database_url,
        seed=seed,
    )

    # ── 1. Ensure data directory exists (SQLite only) ───────────────────────
    if settings.is_sqlite:
        db_path = settings.database_url.split("///")[-1]
        data_dir = Path(db_path).parent
        data_dir.mkdir(parents=True, exist_ok=True)
        logger.info("sqlite_directory_ready", path=str(data_dir))

    # ── 2. Create all tables (idempotent – skips existing) ──────────────────
    try:
        await init_db()
        logger.info("tables_created_or_verified")
    except Exception:
        logger.exception("failed_to_create_tables")
        await dispose_engine()
        sys.exit(1)

    # ── 3. Optional: seed default rules and contexts ────────────────────────
    if seed:
        logger.info("running_seed")
        try:
            from scripts.seed_rules import seed_defaults
            await seed_defaults()
            logger.info("seed_complete")
        except Exception:
            logger.exception("seed_failed")
            await dispose_engine()
            sys.exit(1)

    await dispose_engine()
    logger.info("database_initialization_complete")


if __name__ == "__main__":
    asyncio.run(main())
