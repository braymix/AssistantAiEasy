"""Seed the database with default detection rules and contexts."""

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sqlalchemy import select

from src.config import get_settings
from src.config.logging import setup_logging, get_logger
from src.shared.database import AsyncSessionLocal, init_db, dispose_engine
from src.shared.models import Context, DetectionRule, RuleType

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Seed data
# ---------------------------------------------------------------------------

DEFAULT_CONTEXTS = [
    {"name": "database", "description": "Database configuration and troubleshooting"},
    {"name": "authentication", "description": "Authentication and authorization topics"},
    {"name": "deployment", "description": "Deployment and infrastructure"},
    {"name": "performance", "description": "Performance and optimization"},
    {"name": "errors", "description": "Error codes and stack traces"},
]

DEFAULT_RULES = [
    {
        "name": "database_issues",
        "description": "Detects questions about database configuration and troubleshooting",
        "rule_type": RuleType.COMPOSITE,
        "rule_config": {
            "keywords": ["database", "sql", "query", "migration", "schema", "table", "index"],
            "pattern": r"(?i)(db|database)\s+(error|issue|problem|slow)",
        },
        "target_contexts": ["database"],
        "priority": 10,
    },
    {
        "name": "authentication",
        "description": "Detects authentication and authorization related topics",
        "rule_type": RuleType.COMPOSITE,
        "rule_config": {
            "keywords": ["login", "password", "auth", "token", "jwt", "oauth", "permission", "role"],
            "pattern": r"(?i)(auth|login|sign[- ]?in)\s+(fail|error|issue)",
        },
        "target_contexts": ["authentication"],
        "priority": 10,
    },
    {
        "name": "deployment",
        "description": "Detects deployment and infrastructure questions",
        "rule_type": RuleType.COMPOSITE,
        "rule_config": {
            "keywords": ["deploy", "docker", "kubernetes", "ci/cd", "pipeline", "build", "release"],
            "pattern": r"(?i)(deploy|release|rollback)",
        },
        "target_contexts": ["deployment"],
        "priority": 5,
    },
    {
        "name": "performance",
        "description": "Detects performance-related concerns",
        "rule_type": RuleType.COMPOSITE,
        "rule_config": {
            "keywords": ["slow", "performance", "latency", "timeout", "memory", "cpu", "cache"],
            "pattern": r"(?i)(slow|timeout|high\s+(cpu|memory|latency))",
        },
        "target_contexts": ["performance"],
        "priority": 8,
    },
    {
        "name": "error_codes",
        "description": "Detects error codes and stack traces",
        "rule_type": RuleType.REGEX,
        "rule_config": {
            "pattern": r"(ERR[-_]\d+|HTTP\s+[45]\d{2}|Exception|Traceback)",
        },
        "target_contexts": ["errors"],
        "priority": 9,
    },
]


async def main():
    settings = get_settings()
    setup_logging(settings)

    await init_db()

    async with AsyncSessionLocal() as session:
        # Seed contexts
        for ctx_data in DEFAULT_CONTEXTS:
            existing = await session.execute(
                select(Context).where(Context.name == ctx_data["name"])
            )
            if existing.scalar_one_or_none() is None:
                session.add(Context(**ctx_data))
                logger.info("seeded_context", name=ctx_data["name"])
            else:
                logger.info("context_exists_skipping", name=ctx_data["name"])

        # Seed rules
        for rule_data in DEFAULT_RULES:
            existing = await session.execute(
                select(DetectionRule).where(DetectionRule.name == rule_data["name"])
            )
            if existing.scalar_one_or_none() is None:
                session.add(DetectionRule(**rule_data))
                logger.info("seeded_rule", name=rule_data["name"])
            else:
                logger.info("rule_exists_skipping", name=rule_data["name"])

        await session.commit()

    await dispose_engine()
    logger.info(
        "seeding_complete",
        contexts=len(DEFAULT_CONTEXTS),
        rules=len(DEFAULT_RULES),
    )


if __name__ == "__main__":
    asyncio.run(main())
