"""Admin REST API routes for analytics and dashboard data."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

from fastapi import APIRouter, Depends, Query
from sqlalchemy import String, cast, func, select
from sqlalchemy.ext.asyncio import AsyncSession

from src.admin.schemas.analytics import (
    ContextUsage,
    ConversationTrend,
    DailyCount,
    KnowledgeGrowth,
    OverviewStats,
    RulePerformance,
)
from src.admin.schemas.common import ApiResponse
from src.config.logging import get_logger
from src.shared.database import get_db_session
from src.shared.models import (
    Context,
    Conversation,
    DetectionRule,
    Document,
    KnowledgeItem,
    Message,
)

logger = get_logger(__name__)

router = APIRouter()


# ═══════════════════════════════════════════════════════════════════════════
# OVERVIEW
# ═══════════════════════════════════════════════════════════════════════════


@router.get("/analytics/overview", response_model=ApiResponse[OverviewStats])
async def overview(
    session: AsyncSession = Depends(get_db_session),
) -> ApiResponse[OverviewStats]:
    total_rules = (await session.execute(
        select(func.count()).select_from(DetectionRule)
    )).scalar_one()
    active_rules = (await session.execute(
        select(func.count()).select_from(DetectionRule)
        .where(DetectionRule.enabled.is_(True))
    )).scalar_one()
    total_contexts = (await session.execute(
        select(func.count()).select_from(Context)
    )).scalar_one()
    total_ki = (await session.execute(
        select(func.count()).select_from(KnowledgeItem)
    )).scalar_one()
    verified_ki = (await session.execute(
        select(func.count()).select_from(KnowledgeItem)
        .where(KnowledgeItem.verified.is_(True))
    )).scalar_one()
    total_conv = (await session.execute(
        select(func.count()).select_from(Conversation)
    )).scalar_one()
    total_msg = (await session.execute(
        select(func.count()).select_from(Message)
    )).scalar_one()
    total_docs = (await session.execute(
        select(func.count()).select_from(Document)
    )).scalar_one()

    return ApiResponse(data=OverviewStats(
        total_rules=total_rules,
        active_rules=active_rules,
        total_contexts=total_contexts,
        total_knowledge_items=total_ki,
        verified_knowledge_items=verified_ki,
        pending_knowledge_items=total_ki - verified_ki,
        total_conversations=total_conv,
        total_messages=total_msg,
        total_documents=total_docs,
    ))


# ═══════════════════════════════════════════════════════════════════════════
# CONTEXT USAGE
# ═══════════════════════════════════════════════════════════════════════════


@router.get("/analytics/contexts", response_model=ApiResponse[list[ContextUsage]])
async def context_usage(
    session: AsyncSession = Depends(get_db_session),
) -> ApiResponse[list[ContextUsage]]:
    contexts = (await session.execute(
        select(Context).order_by(Context.name)
    )).scalars().all()

    results: list[ContextUsage] = []
    for ctx in contexts:
        ki_count = (await session.execute(
            select(func.count()).select_from(KnowledgeItem)
            .where(cast(KnowledgeItem.contexts, String).contains(ctx.name))
        )).scalar_one()

        rule_count = (await session.execute(
            select(func.count()).select_from(DetectionRule)
            .where(cast(DetectionRule.target_contexts, String).contains(ctx.name))
        )).scalar_one()

        mention_count = (await session.execute(
            select(func.count()).select_from(Message)
            .where(cast(Message.detected_contexts, String).contains(ctx.name))
        )).scalar_one()

        results.append(ContextUsage(
            context_id=ctx.id,
            context_name=ctx.name,
            knowledge_count=ki_count,
            rule_count=rule_count,
            mention_count=mention_count,
        ))

    return ApiResponse(data=results)


# ═══════════════════════════════════════════════════════════════════════════
# RULE PERFORMANCE
# ═══════════════════════════════════════════════════════════════════════════


@router.get("/analytics/rules", response_model=ApiResponse[list[RulePerformance]])
async def rule_performance(
    session: AsyncSession = Depends(get_db_session),
) -> ApiResponse[list[RulePerformance]]:
    rules = (await session.execute(
        select(DetectionRule).order_by(DetectionRule.priority.desc())
    )).scalars().all()

    return ApiResponse(data=[
        RulePerformance(
            rule_id=r.id,
            rule_name=r.name,
            rule_type=r.rule_type.value if r.rule_type else "keyword",
            enabled=r.enabled,
            priority=r.priority or 0,
            target_contexts=r.target_contexts or [],
        )
        for r in rules
    ])


# ═══════════════════════════════════════════════════════════════════════════
# CONVERSATION TRENDS
# ═══════════════════════════════════════════════════════════════════════════


@router.get("/analytics/conversations", response_model=ApiResponse[ConversationTrend])
async def conversation_trends(
    period_days: int = Query(default=30, ge=1, le=365),
    session: AsyncSession = Depends(get_db_session),
) -> ApiResponse[ConversationTrend]:
    cutoff = datetime.now(timezone.utc) - timedelta(days=period_days)

    total = (await session.execute(
        select(func.count()).select_from(Conversation)
        .where(Conversation.created_at >= cutoff)
    )).scalar_one()

    # Daily breakdown – use date string grouping for cross-DB compat
    rows = (await session.execute(
        select(
            func.date(Conversation.created_at).label("day"),
            func.count().label("cnt"),
        )
        .where(Conversation.created_at >= cutoff)
        .group_by(func.date(Conversation.created_at))
        .order_by(func.date(Conversation.created_at))
    )).all()

    daily = [DailyCount(date=str(r.day), count=r.cnt) for r in rows]

    return ApiResponse(data=ConversationTrend(
        period_days=period_days,
        total=total,
        daily=daily,
    ))


# ═══════════════════════════════════════════════════════════════════════════
# KNOWLEDGE GROWTH
# ═══════════════════════════════════════════════════════════════════════════


@router.get("/analytics/knowledge", response_model=ApiResponse[KnowledgeGrowth])
async def knowledge_growth(
    session: AsyncSession = Depends(get_db_session),
) -> ApiResponse[KnowledgeGrowth]:
    total = (await session.execute(
        select(func.count()).select_from(KnowledgeItem)
    )).scalar_one()
    verified = (await session.execute(
        select(func.count()).select_from(KnowledgeItem)
        .where(KnowledgeItem.verified.is_(True))
    )).scalar_one()

    # Breakdown by content_type
    type_rows = (await session.execute(
        select(KnowledgeItem.content_type, func.count().label("cnt"))
        .group_by(KnowledgeItem.content_type)
    )).all()
    by_type = {
        (r.content_type.value if r.content_type else "unknown"): r.cnt
        for r in type_rows
    }

    # Breakdown by created_by (source)
    source_rows = (await session.execute(
        select(KnowledgeItem.created_by, func.count().label("cnt"))
        .group_by(KnowledgeItem.created_by)
    )).all()
    by_source = {(r.created_by or "unknown"): r.cnt for r in source_rows}

    return ApiResponse(data=KnowledgeGrowth(
        total=total,
        verified=verified,
        pending=total - verified,
        by_type=by_type,
        by_source=by_source,
    ))
