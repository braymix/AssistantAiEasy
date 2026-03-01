"""Database migration helper – wraps Alembic for common operations."""

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import get_settings
from src.config.logging import setup_logging, get_logger
from src.shared.database import init_db, dispose_engine

logger = get_logger(__name__)


async def main():
    settings = get_settings()
    setup_logging(settings)

    action = sys.argv[1] if len(sys.argv) > 1 else "upgrade"

    if action == "upgrade":
        logger.info("running_migration_upgrade")
        await init_db()
        logger.info("migration_complete")
    elif action == "reset":
        logger.info("resetting_database")
        import src.shared.models  # noqa: F401 – register models
        from sqlalchemy.ext.asyncio import create_async_engine
        from src.shared.database import Base

        engine = create_async_engine(settings.database_url)
        async with engine.begin() as conn:
            await conn.run_sync(Base.metadata.drop_all)
            await conn.run_sync(Base.metadata.create_all)
        await engine.dispose()
        logger.info("database_reset_complete")
    else:
        print(f"Unknown action: {action}")
        print("Usage: python scripts/migrate.py [upgrade|reset]")
        sys.exit(1)

    await dispose_engine()


if __name__ == "__main__":
    asyncio.run(main())
