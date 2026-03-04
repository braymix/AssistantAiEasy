"""Admin REST API routes for managing contexts."""

from __future__ import annotations

import uuid

from fastapi import APIRouter, Depends, HTTPException, Query, status
from fastapi.responses import Response
from sqlalchemy import String, cast, func, select
from sqlalchemy.ext.asyncio import AsyncSession

from src.admin.schemas.common import ApiResponse, PaginationMeta
from src.admin.schemas.contexts import (
    ContextCreate,
    ContextKnowledgeItem,
    ContextOut,
    ContextStats,
    ContextUpdate,
)
from src.config.logging import get_logger
from src.shared.database import get_db_session
from src.shared.exceptions import not_found
from src.shared.models import (
    Context,
    DetectionRule,
    KnowledgeItem,
    Message,
)

logger = get_logger(__name__)

router = APIRouter()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _context_to_out(ctx: Context, children: list[ContextOut] | None = None) -> ContextOut:
    return ContextOut(
        id=ctx.id,
        name=ctx.name,
        description=ctx.description,
        parent_id=ctx.parent_id,
        metadata=ctx.metadata_json or {},
        created_at=ctx.created_at,
        children=children or [],
    )


def _build_tree(contexts: list[Context]) -> list[ContextOut]:
    """Build a hierarchical tree from a flat list of contexts."""
    by_id: dict[str, Context] = {c.id: c for c in contexts}
    children_map: dict[str | None, list[Context]] = {}
    for c in contexts:
        children_map.setdefault(c.parent_id, []).append(c)

    def _recurse(parent_id: str | None) -> list[ContextOut]:
        result = []
        for c in children_map.get(parent_id, []):
            kids = _recurse(c.id)
            result.append(_context_to_out(c, kids))
        return result

    return _recurse(None)


# ═══════════════════════════════════════════════════════════════════════════
# LIST
# ═══════════════════════════════════════════════════════════════════════════


@router.get("/contexts", response_model=ApiResponse[list[ContextOut]])
async def list_contexts(
    flat: bool = Query(default=False, description="Return flat list instead of hierarchy"),
    search: str | None = Query(default=None, description="Search by name"),
    limit: int = Query(default=100, ge=1, le=500),
    offset: int = Query(default=0, ge=0),
    session: AsyncSession = Depends(get_db_session),
) -> ApiResponse[list[ContextOut]]:
    q = select(Context)
    if search:
        q = q.where(Context.name.ilike(f"%{search}%"))

    count_q = select(func.count()).select_from(q.subquery())
    total = (await session.execute(count_q)).scalar_one()

    q = q.order_by(Context.name).offset(offset).limit(limit)
    rows = list((await session.execute(q)).scalars().all())

    if flat:
        data = [_context_to_out(c) for c in rows]
    else:
        data = _build_tree(rows)

    return ApiResponse(
        data=data,
        meta=PaginationMeta(total=total, limit=limit, offset=offset),
    )


# ═══════════════════════════════════════════════════════════════════════════
# CREATE
# ═══════════════════════════════════════════════════════════════════════════


@router.post("/contexts", response_model=ApiResponse[ContextOut], status_code=201)
async def create_context(
    payload: ContextCreate,
    session: AsyncSession = Depends(get_db_session),
) -> ApiResponse[ContextOut]:
    if payload.parent_id:
        parent = await session.get(Context, payload.parent_id)
        if parent is None:
            raise not_found(f"Parent context '{payload.parent_id}' not found")

    ctx = Context(
        id=str(uuid.uuid4()),
        name=payload.name,
        description=payload.description,
        parent_id=payload.parent_id,
        metadata_json=payload.metadata,
    )
    session.add(ctx)
    await session.commit()
    await session.refresh(ctx)
    logger.info("context_created", context_id=ctx.id, name=ctx.name)
    return ApiResponse(data=_context_to_out(ctx))


# ═══════════════════════════════════════════════════════════════════════════
# UPDATE
# ═══════════════════════════════════════════════════════════════════════════


@router.put("/contexts/{context_id}", response_model=ApiResponse[ContextOut])
async def update_context(
    context_id: str,
    payload: ContextUpdate,
    session: AsyncSession = Depends(get_db_session),
) -> ApiResponse[ContextOut]:
    ctx = await session.get(Context, context_id)
    if ctx is None:
        raise not_found(f"Context '{context_id}' not found")

    update_data = payload.model_dump(exclude_unset=True)
    if "metadata" in update_data:
        update_data["metadata_json"] = update_data.pop("metadata")
    if "parent_id" in update_data and update_data["parent_id"]:
        parent = await session.get(Context, update_data["parent_id"])
        if parent is None:
            raise not_found(f"Parent context '{update_data['parent_id']}' not found")

    for key, value in update_data.items():
        setattr(ctx, key, value)

    await session.commit()
    await session.refresh(ctx)
    logger.info("context_updated", context_id=ctx.id)
    return ApiResponse(data=_context_to_out(ctx))


# ═══════════════════════════════════════════════════════════════════════════
# DELETE
# ═══════════════════════════════════════════════════════════════════════════


@router.delete("/contexts/{context_id}", status_code=204, response_model=None)
async def delete_context(
    context_id: str,
    session: AsyncSession = Depends(get_db_session),
) -> None:
    ctx = await session.get(Context, context_id)
    if ctx is None:
        raise not_found(f"Context '{context_id}' not found")

    # Check for child contexts
    child_count = (await session.execute(
        select(func.count()).select_from(Context).where(Context.parent_id == context_id)
    )).scalar_one()
    if child_count > 0:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=f"Context has {child_count} child context(s). Remove them first.",
        )

    # Check for rules referencing this context
    rule_count = (await session.execute(
        select(func.count()).select_from(DetectionRule)
        .where(cast(DetectionRule.target_contexts, String).contains(ctx.name))
    )).scalar_one()
    if rule_count > 0:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=f"Context is referenced by {rule_count} rule(s). Update rules first.",
        )

    await session.delete(ctx)
    await session.commit()
    logger.info("context_deleted", context_id=context_id)


# ═══════════════════════════════════════════════════════════════════════════
# KNOWLEDGE items for a context
# ═══════════════════════════════════════════════════════════════════════════


@router.get(
    "/contexts/{context_id}/knowledge",
    response_model=ApiResponse[list[ContextKnowledgeItem]],
)
async def get_context_knowledge(
    context_id: str,
    limit: int = Query(default=20, ge=1, le=100),
    offset: int = Query(default=0, ge=0),
    session: AsyncSession = Depends(get_db_session),
) -> ApiResponse[list[ContextKnowledgeItem]]:
    ctx = await session.get(Context, context_id)
    if ctx is None:
        raise not_found(f"Context '{context_id}' not found")

    q = select(KnowledgeItem).where(
        cast(KnowledgeItem.contexts, String).contains(ctx.name),
    )
    count_q = select(func.count()).select_from(q.subquery())
    total = (await session.execute(count_q)).scalar_one()

    q = q.order_by(KnowledgeItem.created_at.desc()).offset(offset).limit(limit)
    rows = (await session.execute(q)).scalars().all()

    return ApiResponse(
        data=[
            ContextKnowledgeItem(
                id=ki.id,
                content=ki.content,
                content_type=ki.content_type.value if ki.content_type else "manual",
                verified=ki.verified,
                created_at=ki.created_at,
            )
            for ki in rows
        ],
        meta=PaginationMeta(total=total, limit=limit, offset=offset),
    )


# ═══════════════════════════════════════════════════════════════════════════
# STATS for a context
# ═══════════════════════════════════════════════════════════════════════════


@router.get("/contexts/{context_id}/stats", response_model=ApiResponse[ContextStats])
async def get_context_stats(
    context_id: str,
    session: AsyncSession = Depends(get_db_session),
) -> ApiResponse[ContextStats]:
    ctx = await session.get(Context, context_id)
    if ctx is None:
        raise not_found(f"Context '{context_id}' not found")

    knowledge_count = (await session.execute(
        select(func.count()).select_from(KnowledgeItem)
        .where(cast(KnowledgeItem.contexts, String).contains(ctx.name))
    )).scalar_one()

    rule_count = (await session.execute(
        select(func.count()).select_from(DetectionRule)
        .where(cast(DetectionRule.target_contexts, String).contains(ctx.name))
    )).scalar_one()

    conversation_count = (await session.execute(
        select(func.count(func.distinct(Message.conversation_id)))
        .where(cast(Message.detected_contexts, String).contains(ctx.name))
    )).scalar_one()

    return ApiResponse(data=ContextStats(
        context_id=ctx.id,
        context_name=ctx.name,
        knowledge_count=knowledge_count,
        rule_count=rule_count,
        conversation_count=conversation_count,
    ))
