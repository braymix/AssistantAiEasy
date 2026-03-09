"""Seed the database with default detection rules and contexts.

Idempotent: existing rules/contexts are skipped (matched by name).

Can be called standalone or programmatically via ``seed_defaults()``.

Usage:
    python scripts/seed_rules.py
"""

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

# ═══════════════════════════════════════════════════════════════════════════════
# Default contexts
# ═══════════════════════════════════════════════════════════════════════════════

DEFAULT_CONTEXTS = [
    {"name": "projects", "description": "Progetti aziendali e relativi documenti"},
    {"name": "procedures", "description": "Procedure operative e processi interni"},
    {"name": "onboarding", "description": "Onboarding e formazione nuovi dipendenti"},
    {"name": "hr", "description": "Risorse umane, policy e benefit"},
    {"name": "database", "description": "Database configuration and troubleshooting"},
    {"name": "authentication", "description": "Authentication and authorization topics"},
    {"name": "deployment", "description": "Deployment and infrastructure"},
    {"name": "performance", "description": "Performance and optimization"},
    {"name": "errors", "description": "Error codes and stack traces"},
]

# ═══════════════════════════════════════════════════════════════════════════════
# Default detection rules
# ═══════════════════════════════════════════════════════════════════════════════

DEFAULT_RULES = [
    {
        "name": "project_mention",
        "description": "Rileva menzioni di progetti aziendali",
        "rule_type": RuleType.KEYWORD,
        "rule_config": {
            "keywords": ["progetto", "project"],
            "case_sensitive": False,
        },
        "target_contexts": ["projects"],
        "priority": 10,
        "enabled": True,
    },
    {
        "name": "procedure_request",
        "description": "Rileva richieste di procedure",
        "rule_type": RuleType.REGEX,
        "rule_config": {
            "patterns": [
                r"come (si fa|faccio|posso)",
                r"procedura per",
                r"qual è il processo",
            ],
        },
        "target_contexts": ["procedures"],
        "priority": 10,
        "enabled": True,
    },
    {
        "name": "onboarding_topic",
        "description": "Discussioni su onboarding",
        "rule_type": RuleType.SEMANTIC,
        "rule_config": {
            "reference_texts": [
                "nuovo dipendente",
                "primo giorno di lavoro",
                "formazione iniziale",
            ],
            "threshold": 0.75,
        },
        "target_contexts": ["onboarding", "hr"],
        "priority": 8,
        "enabled": True,
    },
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
        "enabled": True,
    },
    {
        "name": "authentication",
        "description": "Detects authentication and authorization related topics",
        "rule_type": RuleType.COMPOSITE,
        "rule_config": {
            "keywords": [
                "login", "password", "auth", "token", "jwt", "oauth", "permission", "role",
            ],
            "pattern": r"(?i)(auth|login|sign[- ]?in)\s+(fail|error|issue)",
        },
        "target_contexts": ["authentication"],
        "priority": 10,
        "enabled": True,
    },
    {
        "name": "deployment",
        "description": "Detects deployment and infrastructure questions",
        "rule_type": RuleType.COMPOSITE,
        "rule_config": {
            "keywords": [
                "deploy", "docker", "kubernetes", "ci/cd", "pipeline", "build", "release",
            ],
            "pattern": r"(?i)(deploy|release|rollback)",
        },
        "target_contexts": ["deployment"],
        "priority": 5,
        "enabled": True,
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
        "enabled": True,
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
        "enabled": True,
    },
]

# Actions mapping (stored separately, used by the trigger system)
DEFAULT_ACTIONS = {
    "project_mention": [
        {"type": "enrich_prompt"},
        {"type": "log_analytics"},
    ],
    "procedure_request": [
        {"type": "enrich_prompt", "max_tokens": 2000},
    ],
    "onboarding_topic": [
        {"type": "enrich_prompt"},
        {"type": "tag_conversation", "tags": ["onboarding"]},
    ],
    "database_issues": [
        {"type": "enrich_prompt"},
    ],
    "authentication": [
        {"type": "enrich_prompt"},
    ],
    "deployment": [
        {"type": "enrich_prompt"},
    ],
    "performance": [
        {"type": "enrich_prompt"},
        {"type": "log_analytics"},
    ],
    "error_codes": [
        {"type": "enrich_prompt"},
        {"type": "log_analytics"},
    ],
}


async def seed_defaults() -> tuple[int, int]:
    """Seed contexts and rules. Returns (contexts_created, rules_created).

    Idempotent: skips items that already exist (by name).
    """
    contexts_created = 0
    rules_created = 0

    async with AsyncSessionLocal() as session:
        # ── Seed contexts ───────────────────────────────────────────────────
        for ctx_data in DEFAULT_CONTEXTS:
            result = await session.execute(
                select(Context).where(Context.name == ctx_data["name"])
            )
            if result.scalar_one_or_none() is None:
                session.add(Context(**ctx_data))
                contexts_created += 1
                logger.info("seeded_context", name=ctx_data["name"])
            else:
                logger.debug("context_exists_skipping", name=ctx_data["name"])

        # ── Seed rules ──────────────────────────────────────────────────────
        for rule_data in DEFAULT_RULES:
            result = await session.execute(
                select(DetectionRule).where(DetectionRule.name == rule_data["name"])
            )
            if result.scalar_one_or_none() is None:
                # Merge actions into rule_config for storage
                rule_copy = dict(rule_data)
                actions = DEFAULT_ACTIONS.get(rule_copy["name"], [])
                if actions:
                    config = dict(rule_copy.get("rule_config") or {})
                    config["actions"] = actions
                    rule_copy["rule_config"] = config
                session.add(DetectionRule(**rule_copy))
                rules_created += 1
                logger.info("seeded_rule", name=rule_data["name"])
            else:
                logger.debug("rule_exists_skipping", name=rule_data["name"])

        await session.commit()

    logger.info(
        "seed_summary",
        contexts_created=contexts_created,
        contexts_total=len(DEFAULT_CONTEXTS),
        rules_created=rules_created,
        rules_total=len(DEFAULT_RULES),
    )
    return contexts_created, rules_created


async def main() -> None:
    settings = get_settings()
    setup_logging(settings)

    logger.info("seeding_database", profile=settings.profile.value)

    await init_db()
    contexts, rules = await seed_defaults()
    await dispose_engine()

    logger.info(
        "seeding_complete",
        contexts=contexts,
        rules=rules,
    )


if __name__ == "__main__":
    asyncio.run(main())
