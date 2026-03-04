"""Pydantic schemas for the /rules admin endpoints."""

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Input
# ---------------------------------------------------------------------------


class RuleCreate(BaseModel):
    """Create a new detection rule."""

    name: str = Field(..., min_length=1, max_length=200)
    description: str | None = None
    rule_type: str = Field(default="keyword", pattern="^(keyword|regex|semantic|composite)$")
    rule_config: dict = Field(default_factory=dict)
    target_contexts: list[str] = Field(default_factory=list)
    priority: int = Field(default=0, ge=0)
    enabled: bool = True


class RuleUpdate(BaseModel):
    """Update an existing detection rule.  All fields optional."""

    name: str | None = Field(default=None, min_length=1, max_length=200)
    description: str | None = None
    rule_type: str | None = Field(default=None, pattern="^(keyword|regex|semantic|composite)$")
    rule_config: dict | None = None
    target_contexts: list[str] | None = None
    priority: int | None = Field(default=None, ge=0)
    enabled: bool | None = None


class RuleTestRequest(BaseModel):
    """Test a detection rule against sample text."""

    text: str = Field(..., min_length=1)


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------


class RuleOut(BaseModel):
    """Detection rule detail."""

    id: str
    name: str
    description: str | None = None
    rule_type: str
    rule_config: dict = Field(default_factory=dict)
    target_contexts: list[str] = Field(default_factory=list)
    priority: int = 0
    enabled: bool = True
    created_at: datetime | None = None
    updated_at: datetime | None = None

    model_config = {"from_attributes": True}


class RuleTestResult(BaseModel):
    """Result of testing a rule against sample text."""

    matched: bool
    confidence: float = 0.0
    extracted: dict = Field(default_factory=dict)
    contexts: list[str] = Field(default_factory=list)
    error: str | None = None


class ReloadResponse(BaseModel):
    """Result of reloading rules into the detection engine."""

    reloaded: bool
    rule_count: int
    message: str
