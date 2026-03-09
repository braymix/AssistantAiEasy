"""
Action registry – maps action type names to action classes and
executes actions for rule matches.

Usage::

    registry = ActionRegistry()
    # Built-in actions are registered automatically.
    # Custom actions can be registered:
    registry.register("my_action", MyAction)

    # Execute all actions configured in a rule_config:
    results = await registry.execute_for_match(match, context, action_configs)

Action configs come from ``DetectionRule.rule_config["actions"]``::

    [
        {"type": "enrich_prompt", "max_tokens": 1500},
        {"type": "tag_conversation", "tags": ["progetto_alpha"]},
        {"type": "log_analytics", "event_type": "context_detected"},
    ]
"""

from __future__ import annotations

import asyncio
import time
from typing import Type

from src.config.logging import get_logger
from src.detection.rules import RuleMatch
from src.detection.triggers import (
    ActionResult,
    BaseTriggerAction,
    ChainAction,
    ConditionalAction,
    EnrichPromptAction,
    LogAnalyticsAction,
    NotifyAdminAction,
    SaveKnowledgeAction,
    TagConversationAction,
    TriggerContext,
)

logger = get_logger(__name__)

# Default retry settings
_DEFAULT_MAX_RETRIES = 0
_DEFAULT_RETRY_DELAY = 1.0


class ActionRegistry:
    """Central registry that maps action type names to their implementations.

    Built-in actions are auto-registered at construction time.
    """

    def __init__(self) -> None:
        self._registry: dict[str, Type[BaseTriggerAction]] = {}
        self._register_builtins()

    # ------------------------------------------------------------------
    # Registration
    # ------------------------------------------------------------------

    def register(self, name: str, action_class: Type[BaseTriggerAction]) -> None:
        """Register an action class under the given name.

        Overwrites any existing registration for the same name.
        """
        self._registry[name] = action_class
        logger.info("action_registered", name=name)

    def get(self, name: str) -> Type[BaseTriggerAction] | None:
        """Look up an action class by name.  Returns ``None`` if unknown."""
        return self._registry.get(name)

    def list_actions(self) -> list[str]:
        """Return all registered action type names."""
        return sorted(self._registry.keys())

    # ------------------------------------------------------------------
    # Execution
    # ------------------------------------------------------------------

    async def execute_for_match(
        self,
        match: RuleMatch,
        context: TriggerContext,
        action_configs: list[dict] | None = None,
        max_retries: int = _DEFAULT_MAX_RETRIES,
        retry_delay: float = _DEFAULT_RETRY_DELAY,
    ) -> list[ActionResult]:
        """Instantiate and execute actions for a rule match.

        Parameters
        ----------
        match : RuleMatch
            The rule match that triggered these actions.
        context : TriggerContext
            Runtime context (DB session, knowledge service, etc.).
        action_configs : list[dict] | None
            List of ``{"type": "<name>", ...config}`` dicts.
            If ``None``, returns an empty list.
        max_retries : int
            Number of retries on failure (0 = no retry).
        retry_delay : float
            Base delay between retries (doubles each attempt).

        Returns
        -------
        list[ActionResult]
            One result per action.  Failed actions produce an error result
            but do not block other actions (error isolation).
        """
        if not action_configs:
            return []

        results: list[ActionResult] = []
        for config in action_configs:
            action_type = config.get("type", "")
            action_cls = self._registry.get(action_type)

            if action_cls is None:
                results.append(ActionResult(
                    action_name=action_type or "unknown",
                    success=False,
                    error=f"unknown action type: {action_type!r}",
                ))
                continue

            # Build config dict (everything except "type")
            action_config = {k: v for k, v in config.items() if k != "type"}

            # Handle special composite types
            if action_type == "chain":
                action = self._build_chain(action_config)
            elif action_type == "conditional":
                action = self._build_conditional(action_config)
            else:
                action = action_cls(config=action_config)

            result = await self._execute_with_retry(
                action, match, context, max_retries, retry_delay,
            )
            results.append(result)

            logger.info(
                "action_executed",
                action=action_type,
                rule=match.rule_name,
                success=result.success,
                duration_ms=result.duration_ms,
            )

        return results

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    async def _execute_with_retry(
        self,
        action: BaseTriggerAction,
        match: RuleMatch,
        context: TriggerContext,
        max_retries: int,
        retry_delay: float,
    ) -> ActionResult:
        """Execute a single action with optional retry on failure."""
        last_result: ActionResult | None = None
        delay = retry_delay

        for attempt in range(max_retries + 1):
            try:
                result = await action.execute(match, context)
                if result.success or attempt >= max_retries:
                    return result
                last_result = result
            except Exception as exc:
                last_result = ActionResult(
                    action_name=action.name,
                    success=False,
                    error=f"exception: {exc}",
                )
                if attempt >= max_retries:
                    return last_result

            logger.warning(
                "action_retry",
                action=action.name,
                attempt=attempt + 1,
                max_retries=max_retries,
                delay=delay,
            )
            await asyncio.sleep(delay)
            delay *= 2

        # Should not reach here, but just in case
        return last_result or ActionResult(
            action_name=action.name, success=False, error="exhausted retries",
        )

    def _build_chain(self, config: dict) -> ChainAction:
        """Build a ChainAction from nested action configs."""
        inner_configs = config.get("actions", [])
        inner_actions: list[BaseTriggerAction] = []
        for ic in inner_configs:
            action_type = ic.get("type", "")
            action_cls = self._registry.get(action_type)
            if action_cls:
                inner_config = {k: v for k, v in ic.items() if k != "type"}
                inner_actions.append(action_cls(config=inner_config))
        return ChainAction(
            actions=inner_actions,
            config={"stop_on_error": config.get("stop_on_error", False)},
        )

    def _build_conditional(self, config: dict) -> ConditionalAction:
        """Build a ConditionalAction from nested action config."""
        inner_config = config.get("action", {})
        action_type = inner_config.get("type", "")
        action_cls = self._registry.get(action_type)
        if action_cls:
            ac = {k: v for k, v in inner_config.items() if k != "type"}
            inner_action = action_cls(config=ac)
        else:
            inner_action = LogAnalyticsAction(config={"event_type": "fallback"})
        return ConditionalAction(
            action=inner_action,
            config={"condition": config.get("condition", "always")},
        )

    def _register_builtins(self) -> None:
        """Register all built-in action types."""
        self._registry["save_knowledge"] = SaveKnowledgeAction
        self._registry["notify_admin"] = NotifyAdminAction
        self._registry["enrich_prompt"] = EnrichPromptAction
        self._registry["tag_conversation"] = TagConversationAction
        self._registry["log_analytics"] = LogAnalyticsAction
        self._registry["chain"] = ChainAction
        self._registry["conditional"] = ConditionalAction


# ---------------------------------------------------------------------------
# Module-level singleton
# ---------------------------------------------------------------------------

_default_registry: ActionRegistry | None = None


def get_action_registry() -> ActionRegistry:
    """Return the default :class:`ActionRegistry` singleton."""
    global _default_registry
    if _default_registry is None:
        _default_registry = ActionRegistry()
    return _default_registry


def reset_action_registry() -> None:
    """Reset the singleton (useful in tests)."""
    global _default_registry
    _default_registry = None
