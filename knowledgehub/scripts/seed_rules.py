"""Seed the database with default detection rules."""

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import get_settings
from src.config.logging import setup_logging, get_logger
from src.knowledge.models import DetectionRule
from src.shared.database import get_db_session, init_db, dispose_engine

import src.knowledge.models  # noqa: F401

logger = get_logger(__name__)

DEFAULT_RULES = [
    {
        "name": "database_issues",
        "description": "Detects questions about database configuration and troubleshooting",
        "keywords": ["database", "sql", "query", "migration", "schema", "table", "index"],
        "pattern": r"(?i)(db|database)\s+(error|issue|problem|slow)",
        "priority": 10,
    },
    {
        "name": "authentication",
        "description": "Detects authentication and authorization related topics",
        "keywords": ["login", "password", "auth", "token", "jwt", "oauth", "permission", "role"],
        "pattern": r"(?i)(auth|login|sign[- ]?in)\s+(fail|error|issue)",
        "priority": 10,
    },
    {
        "name": "deployment",
        "description": "Detects deployment and infrastructure questions",
        "keywords": ["deploy", "docker", "kubernetes", "ci/cd", "pipeline", "build", "release"],
        "pattern": r"(?i)(deploy|release|rollback)",
        "priority": 5,
    },
    {
        "name": "performance",
        "description": "Detects performance-related concerns",
        "keywords": ["slow", "performance", "latency", "timeout", "memory", "cpu", "cache"],
        "pattern": r"(?i)(slow|timeout|high\s+(cpu|memory|latency))",
        "priority": 8,
    },
    {
        "name": "error_codes",
        "description": "Detects error codes and stack traces",
        "keywords": ["error", "exception", "traceback", "stack trace", "500", "404"],
        "pattern": r"(ERR[-_]\d+|HTTP\s+[45]\d{2}|Exception|Traceback)",
        "priority": 9,
    },
]


async def main():
    settings = get_settings()
    setup_logging(settings)

    await init_db()

    from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

    engine = create_async_engine(settings.database_url)
    factory = async_sessionmaker(bind=engine, class_=AsyncSession, expire_on_commit=False)

    async with factory() as session:
        for rule_data in DEFAULT_RULES:
            from sqlalchemy import select

            existing = await session.execute(
                select(DetectionRule).where(DetectionRule.name == rule_data["name"])
            )
            if existing.scalar_one_or_none() is None:
                rule = DetectionRule(**rule_data)
                session.add(rule)
                logger.info("seeded_rule", name=rule_data["name"])
            else:
                logger.info("rule_exists_skipping", name=rule_data["name"])

        await session.commit()

    await engine.dispose()
    await dispose_engine()
    logger.info("seeding_complete", rules=len(DEFAULT_RULES))


if __name__ == "__main__":
    asyncio.run(main())
