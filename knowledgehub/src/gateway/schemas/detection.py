"""Pydantic v2 schemas for the detection endpoints."""

from pydantic import BaseModel, Field


class DetectionRequest(BaseModel):
    text: str = Field(..., min_length=1)
    context: dict = Field(default_factory=dict)


class TriggeredRule(BaseModel):
    rule_id: str
    rule_name: str
    confidence: float = Field(ge=0.0, le=1.0)
    matched_keywords: list[str] = Field(default_factory=list)


class DetectionResult(BaseModel):
    triggered_rules: list[TriggeredRule] = Field(default_factory=list)
    suggested_topics: list[str] = Field(default_factory=list)
    confidence: float = Field(ge=0.0, le=1.0, default=0.0)
    processing_time_ms: int = Field(default=0)
