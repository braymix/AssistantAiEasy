"""
Trigger actions – executed when detection rules match.

Each action implements the :class:`BaseTriggerAction` ABC and is executed
asynchronously with error isolation and optional retry.

Action types:
  1. SaveKnowledgeAction     – extract & persist knowledge
  2. NotifyAdminAction       – log / webhook / email notification
  3. EnrichPromptAction      – inject RAG context into the prompt
  4. TagConversationAction   – tag the current conversation
  5. LogAnalyticsAction      – log event for analytics
  6. ChainAction             – sequential execution of multiple actions
  7. ConditionalAction       – execute only when a condition is met

Legacy exports (:class:`TriggerAction` enum, :class:`Trigger`,
:func:`get_applicable_triggers`) are preserved for backward compatibility.
"""

from __future__ import annotations

import asyncio
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import TYPE_CHECKING, Any

from src.config.logging import get_logger

if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import AsyncSession

    from src.knowledge.service import KnowledgeService

from src.detection.rules import RuleMatch

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class TriggerContext:
    """Runtime context available to every action."""

    session: Any = None  # AsyncSession – Any to avoid import at module level
    knowledge_svc: Any = None  # KnowledgeService
    conversation_id: str | None = None
    user_text: str = ""
    messages: list[dict] = field(default_factory=list)
    metadata: dict = field(default_factory=dict)


@dataclass
class ActionResult:
    """Outcome of a single action execution."""

    action_name: str
    success: bool
    data: dict = field(default_factory=dict)
    error: str | None = None
    duration_ms: int = 0
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


# ---------------------------------------------------------------------------
# Abstract base
# ---------------------------------------------------------------------------


class BaseTriggerAction(ABC):
    """Base class for all trigger actions."""

    def __init__(self, name: str, config: dict | None = None):
        self.name = name
        self.config = config or {}

    @abstractmethod
    async def execute(self, match: RuleMatch, context: TriggerContext) -> ActionResult:
        """Run the action.  Must return an :class:`ActionResult`."""


# ═══════════════════════════════════════════════════════════════════════════
# 1. SaveKnowledgeAction
# ═══════════════════════════════════════════════════════════════════════════


class SaveKnowledgeAction(BaseTriggerAction):
    """Extract and save knowledge from the matched conversation.

    Config keys:
      - ``auto_verify`` (bool, default False): mark item as verified
      - ``min_confidence`` (float, default 0.5): skip if match below this
    """

    def __init__(self, config: dict | None = None):
        super().__init__("save_knowledge", config)

    async def execute(self, match: RuleMatch, context: TriggerContext) -> ActionResult:
        t0 = time.monotonic()
        auto_verify = self.config.get("auto_verify", False)
        min_confidence = self.config.get("min_confidence", 0.5)

        if match.confidence < min_confidence:
            return ActionResult(
                action_name=self.name,
                success=True,
                data={"skipped": True, "reason": "below_min_confidence"},
                duration_ms=_elapsed(t0),
            )

        if context.knowledge_svc is None:
            return ActionResult(
                action_name=self.name,
                success=False,
                error="knowledge_svc not available",
                duration_ms=_elapsed(t0),
            )

        try:
            content = context.user_text or match.extracted.get("reason", "")
            if not content:
                return ActionResult(
                    action_name=self.name,
                    success=True,
                    data={"skipped": True, "reason": "no_content"},
                    duration_ms=_elapsed(t0),
                )

            item = await context.knowledge_svc.add_knowledge(
                content=content,
                contexts=match.contexts,
                source_type="conversation_extract",
                metadata={
                    "rule_name": match.rule_name,
                    "confidence": match.confidence,
                    "auto_verified": auto_verify,
                },
            )

            if auto_verify and context.knowledge_svc is not None:
                await context.knowledge_svc.verify_knowledge(
                    item_id=item.id,
                    verified=True,
                    verified_by="auto",
                )

            logger.info(
                "action_save_knowledge",
                item_id=item.id,
                auto_verify=auto_verify,
            )
            return ActionResult(
                action_name=self.name,
                success=True,
                data={"item_id": item.id, "auto_verified": auto_verify},
                duration_ms=_elapsed(t0),
            )
        except Exception as exc:
            return ActionResult(
                action_name=self.name,
                success=False,
                error=str(exc),
                duration_ms=_elapsed(t0),
            )


# ═══════════════════════════════════════════════════════════════════════════
# 2. NotifyAdminAction
# ═══════════════════════════════════════════════════════════════════════════


class NotifyAdminAction(BaseTriggerAction):
    """Notify administrators via configured channels.

    Config keys:
      - ``channels`` (list[str]): ``["log", "webhook", "email"]``
      - ``template`` (str): message template with ``{rule_name}``,
        ``{confidence}``, ``{contexts}`` placeholders
      - ``webhook_url`` (str): URL for webhook channel
    """

    _DEFAULT_TEMPLATE = (
        "[KnowledgeHub Alert] Rule '{rule_name}' triggered "
        "(confidence={confidence:.2f}, contexts={contexts})"
    )

    def __init__(self, config: dict | None = None):
        super().__init__("notify_admin", config)

    async def execute(self, match: RuleMatch, context: TriggerContext) -> ActionResult:
        t0 = time.monotonic()
        channels = self.config.get("channels", ["log"])
        template = self.config.get("template", self._DEFAULT_TEMPLATE)

        message = template.format(
            rule_name=match.rule_name,
            confidence=match.confidence,
            contexts=", ".join(match.contexts),
        )

        notified: list[str] = []
        errors: list[str] = []

        for channel in channels:
            try:
                if channel == "log":
                    logger.warning("admin_notification", message=message)
                    notified.append("log")
                elif channel == "webhook":
                    await self._send_webhook(message)
                    notified.append("webhook")
                elif channel == "email":
                    # Email placeholder – logs intent
                    logger.info("admin_email_placeholder", message=message)
                    notified.append("email")
                else:
                    errors.append(f"unknown channel: {channel}")
            except Exception as exc:
                errors.append(f"{channel}: {exc}")

        return ActionResult(
            action_name=self.name,
            success=len(errors) == 0,
            data={"notified": notified, "message": message},
            error="; ".join(errors) if errors else None,
            duration_ms=_elapsed(t0),
        )

    async def _send_webhook(self, message: str) -> None:
        """POST notification to webhook URL (placeholder)."""
        webhook_url = self.config.get("webhook_url", "")
        if not webhook_url:
            logger.info("webhook_no_url", message=message)
            return
        # Real implementation would use httpx here
        logger.info("webhook_sent", url=webhook_url, message=message[:200])


# ═══════════════════════════════════════════════════════════════════════════
# 3. EnrichPromptAction
# ═══════════════════════════════════════════════════════════════════════════


class EnrichPromptAction(BaseTriggerAction):
    """Build and inject RAG context into the prompt.

    Config keys:
      - ``max_tokens`` (int, default 2000)
      - ``min_relevance`` (float, default 0.5)
    """

    def __init__(self, config: dict | None = None):
        super().__init__("enrich_prompt", config)

    async def execute(self, match: RuleMatch, context: TriggerContext) -> ActionResult:
        t0 = time.monotonic()
        max_tokens = self.config.get("max_tokens", 2000)
        min_relevance = self.config.get("min_relevance", 0.5)

        if context.knowledge_svc is None:
            return ActionResult(
                action_name=self.name,
                success=False,
                error="knowledge_svc not available",
                duration_ms=_elapsed(t0),
            )

        try:
            rag_context = await context.knowledge_svc.build_rag_context(
                query=context.user_text,
                detected_contexts=match.contexts,
                max_tokens=max_tokens,
            )

            return ActionResult(
                action_name=self.name,
                success=True,
                data={
                    "rag_context": rag_context,
                    "max_tokens": max_tokens,
                    "has_context": bool(rag_context),
                },
                duration_ms=_elapsed(t0),
            )
        except Exception as exc:
            return ActionResult(
                action_name=self.name,
                success=False,
                error=str(exc),
                duration_ms=_elapsed(t0),
            )


# ═══════════════════════════════════════════════════════════════════════════
# 4. TagConversationAction
# ═══════════════════════════════════════════════════════════════════════════


class TagConversationAction(BaseTriggerAction):
    """Add tags to the current conversation's metadata.

    Config keys:
      - ``tags`` (list[str]): tags to add
    """

    def __init__(self, config: dict | None = None):
        super().__init__("tag_conversation", config)

    async def execute(self, match: RuleMatch, context: TriggerContext) -> ActionResult:
        t0 = time.monotonic()
        tags = self.config.get("tags", [])

        if not tags:
            # Fall back to detected contexts as tags
            tags = list(match.contexts)

        if not context.conversation_id:
            return ActionResult(
                action_name=self.name,
                success=False,
                error="no conversation_id in context",
                duration_ms=_elapsed(t0),
            )

        if context.session is None:
            return ActionResult(
                action_name=self.name,
                success=False,
                error="no db session in context",
                duration_ms=_elapsed(t0),
            )

        try:
            from sqlalchemy import select
            from src.shared.models import Conversation

            stmt = select(Conversation).where(
                Conversation.id == context.conversation_id,
            )
            result = await context.session.execute(stmt)
            conv = result.scalar_one_or_none()

            if conv is None:
                return ActionResult(
                    action_name=self.name,
                    success=False,
                    error=f"conversation {context.conversation_id} not found",
                    duration_ms=_elapsed(t0),
                )

            existing_meta = conv.metadata_json or {}
            existing_tags = set(existing_meta.get("tags", []))
            existing_tags.update(tags)
            existing_meta["tags"] = sorted(existing_tags)
            conv.metadata_json = existing_meta
            await context.session.flush()

            logger.info(
                "action_tag_conversation",
                conversation_id=context.conversation_id,
                tags=sorted(existing_tags),
            )
            return ActionResult(
                action_name=self.name,
                success=True,
                data={"conversation_id": context.conversation_id, "tags": sorted(existing_tags)},
                duration_ms=_elapsed(t0),
            )
        except Exception as exc:
            return ActionResult(
                action_name=self.name,
                success=False,
                error=str(exc),
                duration_ms=_elapsed(t0),
            )


# ═══════════════════════════════════════════════════════════════════════════
# 5. LogAnalyticsAction
# ═══════════════════════════════════════════════════════════════════════════


class LogAnalyticsAction(BaseTriggerAction):
    """Log a structured analytics event.

    Config keys:
      - ``event_type`` (str, default "context_detected")
      - ``extra_metadata`` (dict): additional fields to include
    """

    def __init__(self, config: dict | None = None):
        super().__init__("log_analytics", config)

    async def execute(self, match: RuleMatch, context: TriggerContext) -> ActionResult:
        t0 = time.monotonic()
        event_type = self.config.get("event_type", "context_detected")
        extra = self.config.get("extra_metadata", {})

        event = {
            "event_type": event_type,
            "rule_name": match.rule_name,
            "confidence": match.confidence,
            "contexts": match.contexts,
            "conversation_id": context.conversation_id,
            "user_text_length": len(context.user_text),
            **extra,
        }

        logger.info("analytics_event", **event)

        return ActionResult(
            action_name=self.name,
            success=True,
            data=event,
            duration_ms=_elapsed(t0),
        )


# ═══════════════════════════════════════════════════════════════════════════
# 6. ChainAction
# ═══════════════════════════════════════════════════════════════════════════


class ChainAction(BaseTriggerAction):
    """Execute multiple actions in sequence.

    Config keys:
      - ``stop_on_error`` (bool, default False): if True, abort on first failure
    """

    def __init__(
        self,
        actions: list[BaseTriggerAction],
        config: dict | None = None,
    ):
        super().__init__("chain", config)
        self.actions = actions

    async def execute(self, match: RuleMatch, context: TriggerContext) -> ActionResult:
        t0 = time.monotonic()
        stop_on_error = self.config.get("stop_on_error", False)

        results: list[dict] = []
        all_success = True

        for action in self.actions:
            try:
                result = await action.execute(match, context)
                results.append({
                    "action": result.action_name,
                    "success": result.success,
                    "error": result.error,
                    "data": result.data,
                })
                if not result.success:
                    all_success = False
                    if stop_on_error:
                        break
            except Exception as exc:
                results.append({
                    "action": action.name,
                    "success": False,
                    "error": str(exc),
                })
                all_success = False
                if stop_on_error:
                    break

        return ActionResult(
            action_name=self.name,
            success=all_success,
            data={"chain_results": results, "executed": len(results)},
            error=None if all_success else "one or more chain actions failed",
            duration_ms=_elapsed(t0),
        )


# ═══════════════════════════════════════════════════════════════════════════
# 7. ConditionalAction
# ═══════════════════════════════════════════════════════════════════════════


class ConditionalAction(BaseTriggerAction):
    """Execute an action only when a condition is met.

    Config keys:
      - ``condition`` (str): expression evaluated against match/context.
        Supported conditions:
          - ``"confidence >= <float>"``
          - ``"context_contains <name>"``
          - ``"has_conversation"``
          - ``"always"``
    """

    def __init__(
        self,
        action: BaseTriggerAction,
        config: dict | None = None,
    ):
        super().__init__("conditional", config)
        self.action = action

    async def execute(self, match: RuleMatch, context: TriggerContext) -> ActionResult:
        t0 = time.monotonic()
        condition = self.config.get("condition", "always")

        if not _evaluate_condition(condition, match, context):
            return ActionResult(
                action_name=self.name,
                success=True,
                data={"condition": condition, "met": False, "skipped": True},
                duration_ms=_elapsed(t0),
            )

        result = await self.action.execute(match, context)
        return ActionResult(
            action_name=self.name,
            success=result.success,
            data={
                "condition": condition,
                "met": True,
                "inner_action": result.action_name,
                "inner_data": result.data,
            },
            error=result.error,
            duration_ms=_elapsed(t0),
        )


# ---------------------------------------------------------------------------
# Legacy exports (backward compatibility)
# ---------------------------------------------------------------------------


class TriggerType(str, Enum):
    """What happens when a trigger fires (legacy enum)."""

    ENRICH = "enrich"
    ALERT = "alert"
    ROUTE = "route"
    LOG = "log"


# Keep old name as alias
TriggerAction = TriggerType


@dataclass
class Trigger:
    """A trigger ties a detection result to an action (legacy)."""

    name: str
    action: TriggerType
    min_confidence: float = 0.5
    target_topics: list[str] = field(default_factory=list)
    metadata: dict = field(default_factory=dict)


DEFAULT_TRIGGERS: list[Trigger] = [
    Trigger(
        name="knowledge_enrichment",
        action=TriggerType.ENRICH,
        min_confidence=0.3,
        target_topics=["general"],
    ),
    Trigger(
        name="high_confidence_alert",
        action=TriggerType.ALERT,
        min_confidence=0.8,
    ),
    Trigger(
        name="analytics_logger",
        action=TriggerType.LOG,
        min_confidence=0.0,
    ),
]


def get_applicable_triggers(
    confidence: float,
    topics: list[str],
    triggers: list[Trigger] | None = None,
) -> list[Trigger]:
    """Return triggers whose conditions are met by the detection result."""
    triggers = triggers or DEFAULT_TRIGGERS
    applicable: list[Trigger] = []

    for trigger in triggers:
        if confidence < trigger.min_confidence:
            continue
        if trigger.target_topics:
            if not any(t in topics for t in trigger.target_topics):
                continue
        applicable.append(trigger)

    return applicable


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _elapsed(t0: float) -> int:
    return int((time.monotonic() - t0) * 1000)


def _evaluate_condition(condition: str, match: RuleMatch, context: TriggerContext) -> bool:
    """Evaluate a simple condition expression."""
    condition = condition.strip()

    if condition == "always":
        return True

    if condition.startswith("confidence"):
        # e.g. "confidence >= 0.8"
        parts = condition.split()
        if len(parts) == 3:
            op, threshold = parts[1], parts[2]
            try:
                val = float(threshold)
            except ValueError:
                return False
            if op == ">=":
                return match.confidence >= val
            elif op == ">":
                return match.confidence > val
            elif op == "<=":
                return match.confidence <= val
            elif op == "<":
                return match.confidence < val
            elif op == "==":
                return abs(match.confidence - val) < 1e-9

    if condition.startswith("context_contains"):
        # e.g. "context_contains database"
        parts = condition.split(maxsplit=1)
        if len(parts) == 2:
            return parts[1] in match.contexts

    if condition == "has_conversation":
        return context.conversation_id is not None

    logger.warning("unknown_condition", condition=condition)
    return False
