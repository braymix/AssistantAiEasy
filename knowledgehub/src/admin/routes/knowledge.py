"""Admin REST API routes for managing the knowledge base."""

from __future__ import annotations

from datetime import datetime, timezone

from fastapi import APIRouter, Depends, Query
from fastapi.responses import Response
from sqlalchemy import String, cast, func, select
from sqlalchemy.ext.asyncio import AsyncSession

from src.admin.schemas.common import ApiResponse, PaginationMeta
from src.admin.schemas.knowledge import (
    BulkImportRequest,
    BulkImportResponse,
    ExportItem,
    ExportResponse,
    KnowledgeItemOut,
    VerifyAction,
)
from src.config.logging import get_logger
from src.knowledge.service import BulkDocument, KnowledgeService, get_knowledge_service
from src.shared.database import get_db_session
from src.shared.exceptions import not_found
from src.shared.models import ContentType, KnowledgeItem

logger = get_logger(__name__)

router = APIRouter()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _item_to_out(ki: KnowledgeItem) -> KnowledgeItemOut:
    return KnowledgeItemOut(
        id=ki.id,
        content=ki.content,
        content_type=ki.content_type.value if ki.content_type else "manual",
        contexts=ki.contexts or [],
        verified=ki.verified,
        embedding_id=ki.embedding_id,
        source_message_id=ki.source_message_id,
        created_at=ki.created_at,
        created_by=ki.created_by,
    )


# ═══════════════════════════════════════════════════════════════════════════
# LIST
# ═══════════════════════════════════════════════════════════════════════════


@router.get("/knowledge", response_model=ApiResponse[list[KnowledgeItemOut]])
async def list_knowledge(
    verified: bool | None = Query(default=None),
    content_type: str | None = Query(default=None),
    context: str | None = Query(default=None, description="Filter by context name"),
    search: str | None = Query(default=None, description="Search in content"),
    limit: int = Query(default=20, ge=1, le=100),
    offset: int = Query(default=0, ge=0),
    session: AsyncSession = Depends(get_db_session),
) -> ApiResponse[list[KnowledgeItemOut]]:
    q = select(KnowledgeItem)

    if verified is not None:
        q = q.where(KnowledgeItem.verified.is_(verified))
    if content_type is not None:
        q = q.where(KnowledgeItem.content_type == ContentType(content_type))
    if context:
        q = q.where(cast(KnowledgeItem.contexts, String).contains(context))
    if search:
        q = q.where(KnowledgeItem.content.ilike(f"%{search}%"))

    count_q = select(func.count()).select_from(q.subquery())
    total = (await session.execute(count_q)).scalar_one()

    q = q.order_by(KnowledgeItem.created_at.desc()).offset(offset).limit(limit)
    rows = (await session.execute(q)).scalars().all()

    return ApiResponse(
        data=[_item_to_out(ki) for ki in rows],
        meta=PaginationMeta(total=total, limit=limit, offset=offset),
    )


# ═══════════════════════════════════════════════════════════════════════════
# PENDING
# ═══════════════════════════════════════════════════════════════════════════


@router.get("/knowledge/pending", response_model=ApiResponse[list[KnowledgeItemOut]])
async def list_pending(
    limit: int = Query(default=20, ge=1, le=100),
    offset: int = Query(default=0, ge=0),
    session: AsyncSession = Depends(get_db_session),
) -> ApiResponse[list[KnowledgeItemOut]]:
    q = select(KnowledgeItem).where(KnowledgeItem.verified.is_(False))
    count_q = select(func.count()).select_from(q.subquery())
    total = (await session.execute(count_q)).scalar_one()

    q = q.order_by(KnowledgeItem.created_at.desc()).offset(offset).limit(limit)
    rows = (await session.execute(q)).scalars().all()

    return ApiResponse(
        data=[_item_to_out(ki) for ki in rows],
        meta=PaginationMeta(total=total, limit=limit, offset=offset),
    )


# ═══════════════════════════════════════════════════════════════════════════
# VERIFY / REJECT
# ═══════════════════════════════════════════════════════════════════════════


@router.post("/knowledge/{item_id}/verify", response_model=ApiResponse[KnowledgeItemOut])
async def verify_item(
    item_id: str,
    payload: VerifyAction = VerifyAction(),
    service: KnowledgeService = Depends(get_knowledge_service),
) -> ApiResponse[KnowledgeItemOut]:
    item = await service.verify_knowledge(
        item_id=item_id,
        verified=True,
        verified_by=payload.verified_by,
    )
    return ApiResponse(data=_item_to_out(item))


@router.post("/knowledge/{item_id}/reject", response_model=ApiResponse[KnowledgeItemOut])
async def reject_item(
    item_id: str,
    payload: VerifyAction = VerifyAction(),
    service: KnowledgeService = Depends(get_knowledge_service),
) -> ApiResponse[KnowledgeItemOut]:
    item = await service.verify_knowledge(
        item_id=item_id,
        verified=False,
        verified_by=payload.verified_by,
    )
    return ApiResponse(data=_item_to_out(item))


# ═══════════════════════════════════════════════════════════════════════════
# DELETE
# ═══════════════════════════════════════════════════════════════════════════


@router.delete("/knowledge/{item_id}", status_code=204, response_model=None)
async def delete_item(
    item_id: str,
    session: AsyncSession = Depends(get_db_session),
) -> None:
    ki = await session.get(KnowledgeItem, item_id)
    if ki is None:
        raise not_found(f"Knowledge item '{item_id}' not found")
    await session.delete(ki)
    await session.commit()
    logger.info("knowledge_item_deleted", item_id=item_id)


# ═══════════════════════════════════════════════════════════════════════════
# IMPORT
# ═══════════════════════════════════════════════════════════════════════════


@router.post("/knowledge/import", response_model=ApiResponse[BulkImportResponse], status_code=201)
async def bulk_import(
    payload: BulkImportRequest,
    service: KnowledgeService = Depends(get_knowledge_service),
) -> ApiResponse[BulkImportResponse]:
    docs = [
        BulkDocument(content=d.content, metadata=d.metadata, source=d.source)
        for d in payload.documents
    ]
    result = await service.bulk_import(
        documents=docs,
        contexts=payload.contexts,
        chunk_size=payload.chunk_size,
        chunk_overlap=payload.chunk_overlap,
    )
    return ApiResponse(data=BulkImportResponse(
        total=result.total,
        imported=result.imported,
        errors=result.errors,
    ))


# ═══════════════════════════════════════════════════════════════════════════
# EXPORT
# ═══════════════════════════════════════════════════════════════════════════


@router.get("/knowledge/export", response_model=ApiResponse[ExportResponse])
async def export_knowledge(
    verified_only: bool = Query(default=False),
    session: AsyncSession = Depends(get_db_session),
) -> ApiResponse[ExportResponse]:
    q = select(KnowledgeItem)
    if verified_only:
        q = q.where(KnowledgeItem.verified.is_(True))
    q = q.order_by(KnowledgeItem.created_at)

    rows = (await session.execute(q)).scalars().all()

    return ApiResponse(data=ExportResponse(
        items=[
            ExportItem(
                id=ki.id,
                content=ki.content,
                content_type=ki.content_type.value if ki.content_type else "manual",
                contexts=ki.contexts or [],
                verified=ki.verified,
                created_by=ki.created_by,
            )
            for ki in rows
        ],
        total=len(rows),
        exported_at=datetime.now(timezone.utc),
    ))
