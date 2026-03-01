"""Initialize the database – creates all tables.

Tables created:
  conversations, messages, contexts, knowledge_items,
  detection_rules, documents
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


async def main():
    settings = get_settings()
    setup_logging(settings)

    logger.info("initializing_database", profile=settings.profile.value, db=settings.database_url)

    # For SQLite, ensure the directory exists
    if settings.is_sqlite:
        db_path = settings.database_url.split("///")[-1]
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)

    await init_db()
    await dispose_engine()

    logger.info("database_initialized")


if __name__ == "__main__":
    asyncio.run(main())
