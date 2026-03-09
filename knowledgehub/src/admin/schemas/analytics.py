"""Pydantic schemas for the /analytics admin endpoints."""

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, Field


class OverviewStats(BaseModel):
    total_rules: int = 0
    active_rules: int = 0
    total_contexts: int = 0
    total_knowledge_items: int = 0
    verified_knowledge_items: int = 0
    pending_knowledge_items: int = 0
    total_conversations: int = 0
    total_messages: int = 0
    total_documents: int = 0


class ContextUsage(BaseModel):
    context_id: str
    context_name: str
    knowledge_count: int = 0
    rule_count: int = 0
    mention_count: int = 0


class RulePerformance(BaseModel):
    rule_id: str
    rule_name: str
    rule_type: str
    enabled: bool
    priority: int
    target_contexts: list[str] = Field(default_factory=list)


class DailyCount(BaseModel):
    date: str
    count: int


class ConversationTrend(BaseModel):
    period_days: int
    total: int
    daily: list[DailyCount] = Field(default_factory=list)


class KnowledgeGrowth(BaseModel):
    total: int
    verified: int
    pending: int
    by_type: dict[str, int] = Field(default_factory=dict)
    by_source: dict[str, int] = Field(default_factory=dict)
