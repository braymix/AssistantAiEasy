"""Pydantic schemas for the /contexts admin endpoints."""

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Input
# ---------------------------------------------------------------------------


class ContextCreate(BaseModel):
    name: str = Field(..., min_length=1, max_length=200)
    description: str | None = None
    parent_id: str | None = None
    metadata: dict = Field(default_factory=dict)


class ContextUpdate(BaseModel):
    name: str | None = Field(default=None, min_length=1, max_length=200)
    description: str | None = None
    parent_id: str | None = None
    metadata: dict | None = None


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------


class ContextOut(BaseModel):
    id: str
    name: str
    description: str | None = None
    parent_id: str | None = None
    metadata: dict = Field(default_factory=dict)
    created_at: datetime | None = None
    children: list[ContextOut] = Field(default_factory=list)

    model_config = {"from_attributes": True}


class ContextKnowledgeItem(BaseModel):
    id: str
    content: str
    content_type: str
    verified: bool = False
    created_at: datetime | None = None

    model_config = {"from_attributes": True}


class ContextStats(BaseModel):
    context_id: str
    context_name: str
    knowledge_count: int = 0
    rule_count: int = 0
    conversation_count: int = 0
