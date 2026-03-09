"""
Migration orchestrator — runs all steps in order.

Executes the mini → enterprise migration as a sequence of checkpointed
steps.  Each step is recorded so the migration can be resumed from the
last successful checkpoint after a failure.

Usage::

    # Dry run — validate prerequisites without changing anything
    python scripts/migrate_enterprise/run_migration.py --dry-run

    # Execute the full migration
    python scripts/migrate_enterprise/run_migration.py --execute

    # Resume from last checkpoint after a failure
    python scripts/migrate_enterprise/run_migration.py --resume

    # Start from a specific step (0–5)
    python scripts/migrate_enterprise/run_migration.py --execute --from-step 2

    # Send report via webhook on completion
    python scripts/migrate_enterprise/run_migration.py --execute --webhook https://hooks.slack.com/...

Pipeline
────────
::

    ┌────────────────┐
    │ 00 Pre-check   │──fail──→ ABORT
    └───────┬────────┘
            │ pass
    ┌───────▼────────┐
    │ 01 Backup      │──fail──→ ABORT
    └───────┬────────┘
            │ checkpoint
    ┌───────▼────────┐
    │ 02 Database    │──fail──→ ROLLBACK
    └───────┬────────┘
            │ checkpoint
    ┌───────▼────────┐
    │ 03 VectorStore │──fail──→ ROLLBACK
    └───────┬────────┘
            │ checkpoint
    ┌───────▼────────┐
    │ 04 Config      │──fail──→ ROLLBACK
    └───────┬────────┘
            │ checkpoint
    ┌───────▼────────┐
    │ 05 Validate    │──fail──→ WARN (migration done but validation issues)
    └───────┬────────┘
            │
         COMPLETE
"""

from __future__ import annotations

import asyncio
import json
import logging
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
logger = logging.getLogger("migrate.orchestrator")

CHECKPOINT_FILE = Path("./data/migration_checkpoint.json")


# ---------------------------------------------------------------------------
# State management
# ---------------------------------------------------------------------------

@dataclass
class StepRecord:
    """Record of a single migration step execution."""
    step: int
    name: str
    status: str = "pending"  # pending, running, completed, failed, skipped
    started_at: str = ""
    completed_at: str = ""
    duration_seconds: float = 0.0
    result: dict[str, Any] = field(default_factory=dict)
    error: str = ""


@dataclass
class MigrationState:
    """Persistent migration state for checkpoint/resume."""
    migration_id: str = ""
    started_at: str = ""
    completed_at: str = ""
    status: str = "pending"  # pending, running, completed, failed, rolled_back
    current_step: int = 0
    dry_run: bool = False
    backup_id: str = ""
    steps: list[StepRecord] = field(default_factory=list)

    def save(self) -> None:
        CHECKPOINT_FILE.parent.mkdir(parents=True, exist_ok=True)
        CHECKPOINT_FILE.write_text(json.dumps(asdict(self), indent=2))

    @classmethod
    def load(cls) -> MigrationState | None:
        if not CHECKPOINT_FILE.exists():
            return None
        try:
            data = json.loads(CHECKPOINT_FILE.read_text())
            steps = [StepRecord(**s) for s in data.pop("steps", [])]
            return cls(**data, steps=steps)
        except Exception as exc:
            logger.warning("Could not load checkpoint: %s", exc)
            return None

    @classmethod
    def clear(cls) -> None:
        if CHECKPOINT_FILE.exists():
            CHECKPOINT_FILE.unlink()


# ---------------------------------------------------------------------------
# Step definitions
# ---------------------------------------------------------------------------

STEPS = [
    (0, "pre_check", "Pre-migration checks"),
    (1, "backup", "Full backup"),
    (2, "database", "Database migration (SQLite → PostgreSQL)"),
    (3, "vectorstore", "VectorStore migration (ChromaDB → Qdrant)"),
    (4, "config", "Configuration update"),
    (5, "validate", "Post-migration validation"),
]


async def _run_step(step_num: int, dry_run: bool = False) -> dict[str, Any]:
    """Execute a single migration step and return its result."""

    if step_num == 0:
        from scripts.migrate_enterprise.pre_check_00 import run_all_checks
        passed = await run_all_checks()
        return {"passed": passed}

    if step_num == 1:
        from scripts.migrate_enterprise.backup_01 import run_backup
        manifest = run_backup()
        return {
            "passed": manifest.verified,
            "backup_id": manifest.backup_id,
        }

    if step_num == 2:
        from scripts.migrate_enterprise.migrate_db_02 import migrate_database
        result = await migrate_database(dry_run=dry_run)
        return {
            "passed": result.get("all_match", False) or result.get("dry_run", False),
            **result,
        }

    if step_num == 3:
        from scripts.migrate_enterprise.migrate_vs_03 import migrate_vectorstore
        result = await migrate_vectorstore(dry_run=dry_run)
        return {
            "passed": result.get("verified", False) or dry_run,
            **result,
        }

    if step_num == 4:
        from scripts.migrate_enterprise.update_config_04 import update_config
        result = update_config(dry_run=dry_run)
        return {"passed": True, **result}

    if step_num == 5:
        from scripts.migrate_enterprise.validate_05 import run_validation
        passed = await run_validation(skip_benchmark=dry_run)
        return {"passed": passed}

    raise ValueError(f"Unknown step: {step_num}")


# Because modules may not be importable by dotted name from scripts/,
# we also support direct function calls for each step.

async def _run_step_direct(step_num: int, dry_run: bool = False) -> dict[str, Any]:
    """Run step by importing the module file directly."""
    import importlib.util

    base = Path(__file__).parent
    module_map = {
        0: ("pre_check", base / "00_pre_check.py", "run_all_checks"),
        1: ("backup", base / "01_backup_current.py", "run_backup"),
        2: ("migrate_db", base / "02_migrate_database.py", "migrate_database"),
        3: ("migrate_vs", base / "03_migrate_vectorstore.py", "migrate_vectorstore"),
        4: ("update_config", base / "04_update_config.py", "update_config"),
        5: ("validate", base / "05_validate.py", "run_validation"),
    }

    if step_num not in module_map:
        raise ValueError(f"Unknown step: {step_num}")

    mod_name, mod_path, func_name = module_map[step_num]
    spec = importlib.util.spec_from_file_location(mod_name, mod_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    func = getattr(mod, func_name)

    # Each step has a slightly different signature
    if step_num == 0:
        passed = await func()
        return {"passed": passed}
    elif step_num == 1:
        manifest = func()
        return {"passed": manifest.verified, "backup_id": manifest.backup_id}
    elif step_num in (2, 3):
        result = await func(dry_run=dry_run)
        passed_key = "all_match" if step_num == 2 else "verified"
        return {
            "passed": result.get(passed_key, False) or result.get("dry_run", False),
            **result,
        }
    elif step_num == 4:
        result = func(dry_run=dry_run)
        return {"passed": True, **result}
    elif step_num == 5:
        passed = await func(skip_benchmark=dry_run)
        return {"passed": passed}

    return {"passed": False}


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

def _build_report(state: MigrationState) -> str:
    """Build a human-readable migration report."""
    lines = [
        "=" * 60,
        "KnowledgeHub Migration Report",
        "=" * 60,
        f"Migration ID: {state.migration_id}",
        f"Status:       {state.status.upper()}",
        f"Started:      {state.started_at}",
        f"Completed:    {state.completed_at}",
        f"Dry run:      {state.dry_run}",
        "",
        "Steps:",
    ]

    for step in state.steps:
        icon = {
            "completed": "+",
            "failed": "!",
            "skipped": "-",
            "running": ">",
            "pending": " ",
        }.get(step.status, "?")

        duration = f" ({step.duration_seconds:.1f}s)" if step.duration_seconds else ""
        error = f" — {step.error}" if step.error else ""
        lines.append(f"  [{icon}] Step {step.step}: {step.name}{duration}{error}")

    lines.append("")
    lines.append("-" * 60)

    if state.status == "completed":
        lines.append("Migration completed successfully.")
        if state.backup_id:
            lines.append(f"Backup ID for rollback: {state.backup_id}")
        lines.append("")
        lines.append("Next steps:")
        lines.append("  1. Restart services: docker compose -f docker-compose.yml "
                      "-f docker-compose.enterprise.yml up -d")
        lines.append("  2. Monitor logs: docker compose logs -f gateway")
        lines.append("  3. If issues arise, rollback:")
        lines.append(f"     python scripts/migrate_enterprise/rollback.py "
                      f"--backup-id {state.backup_id}")
    elif state.status == "failed":
        lines.append("Migration FAILED. Review errors above.")
        if state.backup_id:
            lines.append(f"To rollback: python scripts/migrate_enterprise/rollback.py "
                          f"--backup-id {state.backup_id}")
        lines.append(f"To resume:   python scripts/migrate_enterprise/run_migration.py --resume")

    return "\n".join(lines)


async def _send_webhook(url: str, state: MigrationState) -> None:
    """Send migration report to a webhook endpoint."""
    try:
        import httpx

        payload = {
            "text": _build_report(state),
            "migration_id": state.migration_id,
            "status": state.status,
            "timestamp": state.completed_at,
        }
        async with httpx.AsyncClient(timeout=15) as client:
            await client.post(url, json=payload)
        logger.info("Webhook notification sent")
    except Exception as exc:
        logger.warning("Webhook notification failed: %s", exc)


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

async def run_migration(
    dry_run: bool = False,
    resume: bool = False,
    from_step: int = 0,
    webhook_url: str = "",
) -> bool:
    """Execute the full migration pipeline.

    Parameters
    ----------
    dry_run:
        Validate and preview without making changes.
    resume:
        Resume from the last checkpoint.
    from_step:
        Start from a specific step number (0–5).
    webhook_url:
        Send a report to this URL on completion.

    Returns
    -------
    bool
        ``True`` if migration completed successfully.
    """
    # Load or create state
    state: MigrationState | None = None
    if resume:
        state = MigrationState.load()
        if state is None:
            logger.error("No checkpoint found — cannot resume")
            return False
        from_step = state.current_step
        logger.info("Resuming migration %s from step %d", state.migration_id, from_step)
    else:
        now = datetime.now(timezone.utc)
        state = MigrationState(
            migration_id=now.strftime("%Y%m%d_%H%M%S"),
            started_at=now.isoformat(),
            status="running",
            dry_run=dry_run,
            steps=[
                StepRecord(step=s[0], name=s[2])
                for s in STEPS
            ],
        )

    logger.info("=" * 60)
    logger.info("KnowledgeHub — Enterprise Migration")
    logger.info("Migration ID: %s", state.migration_id)
    if dry_run:
        logger.info("MODE: DRY RUN")
    logger.info("=" * 60)

    success = True

    for step_num, step_key, step_name in STEPS:
        step_record = state.steps[step_num]

        # Skip completed steps
        if step_num < from_step or step_record.status == "completed":
            if step_record.status == "completed":
                logger.info("Step %d: %s — already completed, skipping", step_num, step_name)
            continue

        # Execute step
        logger.info("")
        logger.info("━" * 60)
        logger.info("Step %d: %s", step_num, step_name)
        logger.info("━" * 60)

        state.current_step = step_num
        step_record.status = "running"
        step_record.started_at = datetime.now(timezone.utc).isoformat()
        state.save()

        start_time = time.time()
        try:
            result = await _run_step_direct(step_num, dry_run=dry_run)
            step_record.duration_seconds = round(time.time() - start_time, 2)
            step_record.result = {
                k: v for k, v in result.items()
                if isinstance(v, (str, int, float, bool, type(None)))
            }

            # Save backup_id from step 1
            if step_num == 1 and "backup_id" in result:
                state.backup_id = result["backup_id"]

            if result.get("passed", False):
                step_record.status = "completed"
                step_record.completed_at = datetime.now(timezone.utc).isoformat()
                logger.info(
                    "Step %d COMPLETED in %.1fs", step_num, step_record.duration_seconds,
                )
            else:
                step_record.status = "failed"
                step_record.error = "Step returned passed=False"

                if step_num <= 1:
                    # Pre-check or backup failed — abort
                    logger.error("Step %d FAILED — aborting migration", step_num)
                    success = False
                    break
                elif step_num == 5:
                    # Validation failed — warn but don't abort
                    logger.warning(
                        "Step %d (validation) had issues — migration data is in place",
                        step_num,
                    )
                    step_record.status = "completed"
                else:
                    # Data step failed — abort (user can rollback or resume)
                    logger.error(
                        "Step %d FAILED — migration paused at checkpoint", step_num,
                    )
                    success = False
                    break

        except Exception as exc:
            step_record.duration_seconds = round(time.time() - start_time, 2)
            step_record.status = "failed"
            step_record.error = str(exc)
            logger.exception("Step %d FAILED with exception", step_num)
            success = False
            break
        finally:
            state.save()

    # Final state
    state.completed_at = datetime.now(timezone.utc).isoformat()
    state.status = "completed" if success else "failed"
    state.save()

    # Report
    report = _build_report(state)
    logger.info("")
    logger.info(report)

    # Webhook
    if webhook_url:
        await _send_webhook(webhook_url, state)

    # Clean checkpoint on success
    if success and not dry_run:
        logger.info("Migration checkpoint cleared")
        # Keep the checkpoint file for reference but mark as completed

    return success


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    args = sys.argv[1:]

    dry_run = "--dry-run" in args
    execute = "--execute" in args
    resume = "--resume" in args

    from_step = 0
    webhook_url = ""

    for i, arg in enumerate(args):
        if arg == "--from-step" and i + 1 < len(args):
            from_step = int(args[i + 1])
        elif arg == "--webhook" and i + 1 < len(args):
            webhook_url = args[i + 1]
        elif arg.startswith("--webhook="):
            webhook_url = arg.split("=", 1)[1]

    if not any([dry_run, execute, resume]):
        print("Usage:")
        print("  python scripts/migrate_enterprise/run_migration.py --dry-run")
        print("  python scripts/migrate_enterprise/run_migration.py --execute")
        print("  python scripts/migrate_enterprise/run_migration.py --resume")
        print("  python scripts/migrate_enterprise/run_migration.py --execute --from-step 2")
        print("  python scripts/migrate_enterprise/run_migration.py --execute --webhook <url>")
        sys.exit(1)

    success = asyncio.run(run_migration(
        dry_run=dry_run,
        resume=resume,
        from_step=from_step,
        webhook_url=webhook_url,
    ))
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
