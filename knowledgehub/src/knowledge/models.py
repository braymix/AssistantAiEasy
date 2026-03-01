"""Backward-compatible re-exports – canonical models live in src.shared.models."""

from src.shared.models import (
    ContentType,
    Context,
    Conversation,
    DetectionRule,
    Document,
    KnowledgeItem,
    Message,
    MessageRole,
    RuleType,
)

__all__ = [
    "ContentType",
    "Context",
    "Conversation",
    "DetectionRule",
    "Document",
    "KnowledgeItem",
    "Message",
    "MessageRole",
    "RuleType",
]
