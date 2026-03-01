"""Pydantic v2 schemas for the knowledge endpoints."""

from datetime import datetime

from pydantic import BaseModel, Field


class DocumentCreate(BaseModel):
    title: str = Field(..., min_length=1, max_length=500)
    content: str = Field(..., min_length=1)
    metadata: dict = Field(default_factory=dict)


class DocumentOut(BaseModel):
    id: str
    title: str
    content: str
    metadata: dict
    chunk_count: int = 0
    created_at: datetime
    updated_at: datetime

    model_config = {"from_attributes": True}


class DocumentList(BaseModel):
    items: list[DocumentOut]
    total: int
