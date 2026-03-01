"""Knowledge base endpoints – document CRUD + search + manual add + contexts."""

from fastapi import APIRouter, Depends, UploadFile
from pydantic import BaseModel, Field
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from src.gateway.schemas.knowledge import (
    DocumentCreate,
    DocumentList,
    DocumentOut,
)
from src.knowledge.service import KnowledgeService
from src.shared.database import get_db_session
from src.shared.models import ContentType, Context, KnowledgeItem

router = APIRouter()


def _service(session: AsyncSession = Depends(get_db_session)) -> KnowledgeService:
    return KnowledgeService(session)


# ═══════════════════════════════════════════════════════════════════════════
# Document CRUD (existing)
# ═══════════════════════════════════════════════════════════════════════════

@router.post("/documents", response_model=DocumentOut, status_code=201)
async def create_document(
    payload: DocumentCreate,
    service: KnowledgeService = Depends(_service),
) -> DocumentOut:
    doc = await service.add_document(
        title=payload.title,
        content=payload.content,
        metadata=payload.metadata,
    )
    return doc


@router.get("/documents", response_model=DocumentList)
async def list_documents(
    skip: int = 0,
    limit: int = 20,
    service: KnowledgeService = Depends(_service),
) -> DocumentList:
    docs, total = await service.list_documents(skip=skip, limit=limit)
    return DocumentList(items=docs, total=total)


@router.get("/documents/{document_id}", response_model=DocumentOut)
async def get_document(
    document_id: str,
    service: KnowledgeService = Depends(_service),
) -> DocumentOut:
    return await service.get_document(document_id)


@router.delete("/documents/{document_id}", status_code=204)
async def delete_document(
    document_id: str,
    service: KnowledgeService = Depends(_service),
) -> None:
    await service.delete_document(document_id)


@router.post("/documents/upload", response_model=DocumentOut, status_code=201)
async def upload_document(
    file: UploadFile,
    service: KnowledgeService = Depends(_service),
) -> DocumentOut:
    content = (await file.read()).decode("utf-8")
    doc = await service.add_document(
        title=file.filename or "untitled",
        content=content,
        metadata={"source": "upload", "filename": file.filename},
    )
    return doc


# ═══════════════════════════════════════════════════════════════════════════
# Semantic search
# ═══════════════════════════════════════════════════════════════════════════

class SearchRequest(BaseModel):
    query: str = Field(..., min_length=1)
    top_k: int = Field(default=5, ge=1, le=50)
    contexts: list[str] = Field(default_factory=list)


class SearchResultItem(BaseModel):
    id: str
    content: str
    title: str
    score: float
    metadata: dict = Field(default_factory=dict)


class SearchResponse(BaseModel):
    results: list[SearchResultItem]
    query: str


@router.post("/search", response_model=SearchResponse)
async def search_knowledge(
    payload: SearchRequest,
    service: KnowledgeService = Depends(_service),
) -> SearchResponse:
    """Semantic search across the knowledge base."""
    results = await service.search(payload.query, top_k=payload.top_k)
    return SearchResponse(
        query=payload.query,
        results=[
            SearchResultItem(
                id=r.id,
                content=r.content,
                title=r.title,
                score=r.score,
                metadata=r.metadata,
            )
            for r in results
        ],
    )


# ═══════════════════════════════════════════════════════════════════════════
# Manual knowledge add
# ═══════════════════════════════════════════════════════════════════════════

class KnowledgeAddRequest(BaseModel):
    content: str = Field(..., min_length=1)
    content_type: str = Field(default="manual")
    contexts: list[str] = Field(default_factory=list)
    created_by: str = Field(default="api")


class KnowledgeAddResponse(BaseModel):
    id: str
    content: str
    content_type: str
    contexts: list[str]
    verified: bool
    created_by: str | None

    model_config = {"from_attributes": True}


@router.post("/add", response_model=KnowledgeAddResponse, status_code=201)
async def add_knowledge(
    payload: KnowledgeAddRequest,
    session: AsyncSession = Depends(get_db_session),
) -> KnowledgeAddResponse:
    """Manually add a knowledge item."""
    try:
        ct = ContentType(payload.content_type)
    except ValueError:
        ct = ContentType.MANUAL

    item = KnowledgeItem(
        content=payload.content,
        content_type=ct,
        contexts=payload.contexts,
        verified=False,
        created_by=payload.created_by,
    )
    session.add(item)
    await session.flush()

    return KnowledgeAddResponse(
        id=item.id,
        content=item.content,
        content_type=item.content_type.value,
        contexts=item.contexts or [],
        verified=item.verified,
        created_by=item.created_by,
    )


# ═══════════════════════════════════════════════════════════════════════════
# Context listing
# ═══════════════════════════════════════════════════════════════════════════

class ContextOut(BaseModel):
    id: str
    name: str
    description: str | None
    parent_id: str | None
    metadata: dict = Field(default_factory=dict)

    model_config = {"from_attributes": True}


class ContextListResponse(BaseModel):
    items: list[ContextOut]
    total: int


@router.get("/contexts", response_model=ContextListResponse)
async def list_contexts(
    session: AsyncSession = Depends(get_db_session),
) -> ContextListResponse:
    """List all available contexts."""
    result = await session.execute(select(Context).order_by(Context.name))
    contexts = list(result.scalars().all())
    return ContextListResponse(
        items=[
            ContextOut(
                id=c.id,
                name=c.name,
                description=c.description,
                parent_id=c.parent_id,
                metadata=c.metadata_json or {},
            )
            for c in contexts
        ],
        total=len(contexts),
    )
