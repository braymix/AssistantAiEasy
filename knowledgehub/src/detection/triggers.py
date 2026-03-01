"""Trigger definitions for context detection."""

from dataclasses import dataclass, field
from enum import Enum


class TriggerAction(str, Enum):
    """What happens when a trigger fires."""

    ENRICH = "enrich"       # Enrich context with additional knowledge
    ALERT = "alert"         # Send an alert / notification
    ROUTE = "route"         # Route to a specific handler
    LOG = "log"             # Log the event for analytics


@dataclass
class Trigger:
    """A trigger ties a detection result to an action."""

    name: str
    action: TriggerAction
    min_confidence: float = 0.5
    target_topics: list[str] = field(default_factory=list)
    metadata: dict = field(default_factory=dict)


# Default triggers that ship with the system
DEFAULT_TRIGGERS: list[Trigger] = [
    Trigger(
        name="knowledge_enrichment",
        action=TriggerAction.ENRICH,
        min_confidence=0.3,
        target_topics=["general"],
    ),
    Trigger(
        name="high_confidence_alert",
        action=TriggerAction.ALERT,
        min_confidence=0.8,
    ),
    Trigger(
        name="analytics_logger",
        action=TriggerAction.LOG,
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
        # If trigger has target_topics, at least one must match
        if trigger.target_topics:
            if not any(t in topics for t in trigger.target_topics):
                continue
        applicable.append(trigger)

    return applicable
