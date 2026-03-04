"""Pydantic schemas for the /knowledge admin endpoints."""

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------


class KnowledgeItemOut(BaseModel):
    id: str
    content: str
    content_type: str
    contexts: list[str] = Field(default_factory=list)
    verified: bool = False
    embedding_id: str | None = None
    source_message_id: str | None = None
    created_at: datetime | None = None
    created_by: str | None = None

    model_config = {"from_attributes": True}


# ---------------------------------------------------------------------------
# Input
# ---------------------------------------------------------------------------


class VerifyAction(BaseModel):
    verified_by: str = Field(default="admin")


class BulkDocumentInput(BaseModel):
    content: str = Field(..., min_length=1)
    metadata: dict = Field(default_factory=dict)
    source: str = Field(default="import")


class BulkImportRequest(BaseModel):
    documents: list[BulkDocumentInput] = Field(..., min_length=1)
    contexts: list[str] = Field(default_factory=list)
    chunk_size: int | None = None
    chunk_overlap: int | None = None


class BulkImportResponse(BaseModel):
    total: int
    imported: int
    errors: list[str] = Field(default_factory=list)


class ExportItem(BaseModel):
    id: str
    content: str
    content_type: str
    contexts: list[str] = Field(default_factory=list)
    verified: bool
    created_by: str | None = None


class ExportResponse(BaseModel):
    items: list[ExportItem]
    total: int
    exported_at: datetime
