"""Knowledge base CRUD endpoints."""

from fastapi import APIRouter, Depends, UploadFile
from sqlalchemy.ext.asyncio import AsyncSession

from src.gateway.schemas.knowledge import (
    DocumentCreate,
    DocumentOut,
    DocumentList,
)
from src.knowledge.service import KnowledgeService
from src.shared.database import get_db_session

router = APIRouter()


def _service(session: AsyncSession = Depends(get_db_session)) -> KnowledgeService:
    return KnowledgeService(session)


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
