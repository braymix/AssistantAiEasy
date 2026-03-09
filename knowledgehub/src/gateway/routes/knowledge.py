"""Knowledge base endpoints – document CRUD + search + knowledge management."""

from fastapi import APIRouter, Depends, UploadFile
from pydantic import BaseModel, Field
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from src.gateway.schemas.knowledge import (
    DocumentCreate,
    DocumentList,
    DocumentOut,
)
from src.knowledge.service import (
    BulkDocument,
    KnowledgeService,
    get_knowledge_service,
)
from src.shared.database import get_db_session
from src.shared.models import Context

router = APIRouter()


# ═══════════════════════════════════════════════════════════════════════════
# Document CRUD
# ═══════════════════════════════════════════════════════════════════════════

@router.post("/documents", response_model=DocumentOut, status_code=201)
async def create_document(
    payload: DocumentCreate,
    service: KnowledgeService = Depends(get_knowledge_service),
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
    service: KnowledgeService = Depends(get_knowledge_service),
) -> DocumentList:
    docs, total = await service.list_documents(skip=skip, limit=limit)
    return DocumentList(items=docs, total=total)


@router.get("/documents/{document_id}", response_model=DocumentOut)
async def get_document(
    document_id: str,
    service: KnowledgeService = Depends(get_knowledge_service),
) -> DocumentOut:
    return await service.get_document(document_id)


@router.delete("/documents/{document_id}", status_code=204)
async def delete_document(
    document_id: str,
    service: KnowledgeService = Depends(get_knowledge_service),
) -> None:
    await service.delete_document(document_id)


@router.post("/documents/upload", response_model=DocumentOut, status_code=201)
async def upload_document(
    file: UploadFile,
    service: KnowledgeService = Depends(get_knowledge_service),
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
    min_score: float = Field(default=0.7, ge=0.0, le=1.0)


class SearchResultItem(BaseModel):
    id: str
    content: str
    score: float
    contexts: list[str] = Field(default_factory=list)
    verified: bool = False
    metadata: dict = Field(default_factory=dict)


class SearchResponse(BaseModel):
    results: list[SearchResultItem]
    query: str


@router.post("/search", response_model=SearchResponse)
async def search_knowledge(
    payload: SearchRequest,
    service: KnowledgeService = Depends(get_knowledge_service),
) -> SearchResponse:
    """Semantic search across the knowledge base with context filtering."""
    results = await service.search_knowledge(
        query=payload.query,
        contexts=payload.contexts if payload.contexts else None,
        n_results=payload.top_k,
        min_score=payload.min_score,
    )
    return SearchResponse(
        query=payload.query,
        results=[
            SearchResultItem(
                id=r.item.id,
                content=r.item.content,
                score=r.score,
                contexts=r.item.contexts or [],
                verified=r.item.verified,
                metadata={"highlights": r.highlights},
            )
            for r in results
        ],
    )


# ═══════════════════════════════════════════════════════════════════════════
# Knowledge add
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
    service: KnowledgeService = Depends(get_knowledge_service),
) -> KnowledgeAddResponse:
    """Add a knowledge item (embeds and stores in vector store)."""
    item = await service.add_knowledge(
        content=payload.content,
        contexts=payload.contexts,
        source_type=payload.content_type,
        metadata={"created_by": payload.created_by},
    )
    return KnowledgeAddResponse(
        id=item.id,
        content=item.content,
        content_type=item.content_type.value,
        contexts=item.contexts or [],
        verified=item.verified,
        created_by=item.created_by,
    )


# ═══════════════════════════════════════════════════════════════════════════
# Knowledge extraction from conversation
# ═══════════════════════════════════════════════════════════════════════════

class ExtractionRequest(BaseModel):
    conversation_id: str
    force: bool = False


class ExtractionResponse(BaseModel):
    conversation_id: str
    items_extracted: int
    items: list[KnowledgeAddResponse]


@router.post("/extract", response_model=ExtractionResponse, status_code=201)
async def extract_from_conversation(
    payload: ExtractionRequest,
    service: KnowledgeService = Depends(get_knowledge_service),
) -> ExtractionResponse:
    """Extract knowledge items from a conversation using LLM."""
    items = await service.extract_knowledge_from_conversation(
        conversation_id=payload.conversation_id,
        force=payload.force,
    )
    return ExtractionResponse(
        conversation_id=payload.conversation_id,
        items_extracted=len(items),
        items=[
            KnowledgeAddResponse(
                id=item.id,
                content=item.content,
                content_type=item.content_type.value,
                contexts=item.contexts or [],
                verified=item.verified,
                created_by=item.created_by,
            )
            for item in items
        ],
    )


# ═══════════════════════════════════════════════════════════════════════════
# Knowledge verification (admin)
# ═══════════════════════════════════════════════════════════════════════════

class VerifyRequest(BaseModel):
    verified: bool
    verified_by: str


class VerifyResponse(BaseModel):
    id: str
    verified: bool
    verified_by: str | None

    model_config = {"from_attributes": True}


@router.put("/items/{item_id}/verify", response_model=VerifyResponse)
async def verify_knowledge(
    item_id: str,
    payload: VerifyRequest,
    service: KnowledgeService = Depends(get_knowledge_service),
) -> VerifyResponse:
    """Admin approves or rejects a knowledge item."""
    item = await service.verify_knowledge(
        item_id=item_id,
        verified=payload.verified,
        verified_by=payload.verified_by,
    )
    return VerifyResponse(
        id=item.id,
        verified=item.verified,
        verified_by=item.created_by,
    )


# ═══════════════════════════════════════════════════════════════════════════
# Bulk import
# ═══════════════════════════════════════════════════════════════════════════

class BulkDocumentInput(BaseModel):
    content: str = Field(..., min_length=1)
    metadata: dict = Field(default_factory=dict)
    source: str = Field(default="import")


class BulkImportRequest(BaseModel):
    documents: list[BulkDocumentInput]
    contexts: list[str] = Field(default_factory=list)
    chunk_size: int | None = None
    chunk_overlap: int | None = None


class BulkImportResponse(BaseModel):
    total: int
    imported: int
    errors: list[str]


@router.post("/bulk-import", response_model=BulkImportResponse, status_code=201)
async def bulk_import(
    payload: BulkImportRequest,
    service: KnowledgeService = Depends(get_knowledge_service),
) -> BulkImportResponse:
    """Bulk-import documents: chunk, embed, and store."""
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
    return BulkImportResponse(
        total=result.total,
        imported=result.imported,
        errors=result.errors,
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
