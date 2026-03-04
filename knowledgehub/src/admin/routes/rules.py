"""Admin REST API routes for managing detection rules."""

from __future__ import annotations

import uuid

from fastapi import APIRouter, Depends, Query
from fastapi.responses import Response
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from src.admin.schemas.common import ApiResponse, PaginationMeta
from src.admin.schemas.rules import (
    ReloadResponse,
    RuleCreate,
    RuleOut,
    RuleTestRequest,
    RuleTestResult,
    RuleUpdate,
)
from src.config.logging import get_logger
from src.detection.engine import DetectionEngine, _convert_db_rules
from src.detection.rules import DetectionContext
from src.shared.database import get_db_session
from src.shared.exceptions import not_found
from src.shared.models import DetectionRule, RuleType

logger = get_logger(__name__)

router = APIRouter()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _rule_to_out(rule: DetectionRule) -> RuleOut:
    return RuleOut(
        id=rule.id,
        name=rule.name,
        description=rule.description,
        rule_type=rule.rule_type.value if rule.rule_type else "keyword",
        rule_config=rule.rule_config or {},
        target_contexts=rule.target_contexts or [],
        priority=rule.priority or 0,
        enabled=rule.enabled,
        created_at=rule.created_at,
        updated_at=rule.updated_at,
    )


# ═══════════════════════════════════════════════════════════════════════════
# LIST
# ═══════════════════════════════════════════════════════════════════════════


@router.get("/rules", response_model=ApiResponse[list[RuleOut]])
async def list_rules(
    rule_type: str | None = Query(default=None, description="Filter by rule type"),
    enabled: bool | None = Query(default=None, description="Filter by enabled status"),
    search: str | None = Query(default=None, description="Search by name"),
    limit: int = Query(default=20, ge=1, le=100),
    offset: int = Query(default=0, ge=0),
    session: AsyncSession = Depends(get_db_session),
) -> ApiResponse[list[RuleOut]]:
    q = select(DetectionRule)

    if rule_type is not None:
        q = q.where(DetectionRule.rule_type == RuleType(rule_type))
    if enabled is not None:
        q = q.where(DetectionRule.enabled.is_(enabled))
    if search:
        q = q.where(DetectionRule.name.ilike(f"%{search}%"))

    # Count
    count_q = select(func.count()).select_from(q.subquery())
    total = (await session.execute(count_q)).scalar_one()

    # Fetch page
    q = q.order_by(DetectionRule.priority.desc()).offset(offset).limit(limit)
    rows = (await session.execute(q)).scalars().all()

    return ApiResponse(
        data=[_rule_to_out(r) for r in rows],
        meta=PaginationMeta(total=total, limit=limit, offset=offset),
    )


# ═══════════════════════════════════════════════════════════════════════════
# GET
# ═══════════════════════════════════════════════════════════════════════════


@router.get("/rules/{rule_id}", response_model=ApiResponse[RuleOut])
async def get_rule(
    rule_id: str,
    session: AsyncSession = Depends(get_db_session),
) -> ApiResponse[RuleOut]:
    rule = await session.get(DetectionRule, rule_id)
    if rule is None:
        raise not_found(f"Rule '{rule_id}' not found")
    return ApiResponse(data=_rule_to_out(rule))


# ═══════════════════════════════════════════════════════════════════════════
# CREATE
# ═══════════════════════════════════════════════════════════════════════════


@router.post("/rules", response_model=ApiResponse[RuleOut], status_code=201)
async def create_rule(
    payload: RuleCreate,
    session: AsyncSession = Depends(get_db_session),
) -> ApiResponse[RuleOut]:
    rule = DetectionRule(
        id=str(uuid.uuid4()),
        name=payload.name,
        description=payload.description,
        rule_type=RuleType(payload.rule_type),
        rule_config=payload.rule_config,
        target_contexts=payload.target_contexts,
        priority=payload.priority,
        enabled=payload.enabled,
    )
    session.add(rule)
    await session.commit()
    await session.refresh(rule)
    logger.info("rule_created", rule_id=rule.id, name=rule.name)
    return ApiResponse(data=_rule_to_out(rule))


# ═══════════════════════════════════════════════════════════════════════════
# UPDATE
# ═══════════════════════════════════════════════════════════════════════════


@router.put("/rules/{rule_id}", response_model=ApiResponse[RuleOut])
async def update_rule(
    rule_id: str,
    payload: RuleUpdate,
    session: AsyncSession = Depends(get_db_session),
) -> ApiResponse[RuleOut]:
    rule = await session.get(DetectionRule, rule_id)
    if rule is None:
        raise not_found(f"Rule '{rule_id}' not found")

    update_data = payload.model_dump(exclude_unset=True)
    if "rule_type" in update_data and update_data["rule_type"] is not None:
        update_data["rule_type"] = RuleType(update_data["rule_type"])

    for key, value in update_data.items():
        setattr(rule, key, value)

    await session.commit()
    await session.refresh(rule)
    logger.info("rule_updated", rule_id=rule.id)
    return ApiResponse(data=_rule_to_out(rule))


# ═══════════════════════════════════════════════════════════════════════════
# DELETE
# ═══════════════════════════════════════════════════════════════════════════


@router.delete("/rules/{rule_id}", status_code=204, response_model=None)
async def delete_rule(
    rule_id: str,
    session: AsyncSession = Depends(get_db_session),
) -> None:
    rule = await session.get(DetectionRule, rule_id)
    if rule is None:
        raise not_found(f"Rule '{rule_id}' not found")
    await session.delete(rule)
    await session.commit()
    logger.info("rule_deleted", rule_id=rule_id)


# ═══════════════════════════════════════════════════════════════════════════
# TEST
# ═══════════════════════════════════════════════════════════════════════════


@router.post("/rules/{rule_id}/test", response_model=ApiResponse[RuleTestResult])
async def test_rule(
    rule_id: str,
    payload: RuleTestRequest,
    session: AsyncSession = Depends(get_db_session),
) -> ApiResponse[RuleTestResult]:
    """Test a single rule against sample text without side effects."""
    db_rule = await session.get(DetectionRule, rule_id)
    if db_rule is None:
        raise not_found(f"Rule '{rule_id}' not found")

    converted = _convert_db_rules([db_rule])
    if not converted:
        return ApiResponse(data=RuleTestResult(
            matched=False,
            error="Rule could not be converted (missing config?)",
        ))

    rule_obj = converted[0]
    try:
        match = await rule_obj.match(payload.text, DetectionContext())
        if match is not None:
            return ApiResponse(data=RuleTestResult(
                matched=True,
                confidence=match.confidence,
                extracted=match.extracted,
                contexts=match.contexts,
            ))
        return ApiResponse(data=RuleTestResult(matched=False))
    except Exception as exc:
        return ApiResponse(data=RuleTestResult(matched=False, error=str(exc)))


# ═══════════════════════════════════════════════════════════════════════════
# RELOAD
# ═══════════════════════════════════════════════════════════════════════════


@router.post("/rules/reload", response_model=ApiResponse[ReloadResponse])
async def reload_rules(
    session: AsyncSession = Depends(get_db_session),
) -> ApiResponse[ReloadResponse]:
    """Reload all rules into the detection engine from the database."""
    engine = DetectionEngine(session)
    await engine.load_rules_from_db()
    rule_count = len(engine._injected_rules)
    logger.info("rules_reloaded_via_admin", count=rule_count)
    return ApiResponse(data=ReloadResponse(
        reloaded=True,
        rule_count=rule_count,
        message=f"Successfully reloaded {rule_count} rules",
    ))
