"""Tests for the ActionRegistry – registration, lookup, execution & retry."""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest

from src.detection.action_registry import (
    ActionRegistry,
    get_action_registry,
    reset_action_registry,
)
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


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def registry():
    return ActionRegistry()


@pytest.fixture
def rule_match():
    return RuleMatch(
        rule_name="test_rule",
        confidence=0.85,
        extracted={"matched_keywords": ["database"]},
        contexts=["tech", "database"],
    )


@pytest.fixture
def trigger_context():
    return TriggerContext(
        user_text="How to configure database?",
        conversation_id="conv-123",
    )


# ═══════════════════════════════════════════════════════════════════════════
# Registration & lookup
# ═══════════════════════════════════════════════════════════════════════════


def test_builtins_registered(registry):
    """All 7 built-in action types should be registered at init."""
    actions = registry.list_actions()
    for expected in [
        "chain", "conditional", "enrich_prompt", "log_analytics",
        "notify_admin", "save_knowledge", "tag_conversation",
    ]:
        assert expected in actions


def test_get_known_action(registry):
    cls = registry.get("log_analytics")
    assert cls is LogAnalyticsAction


def test_get_unknown_action(registry):
    assert registry.get("nonexistent") is None


def test_register_custom_action(registry):

    class MyAction(BaseTriggerAction):
        def __init__(self, config=None):
            super().__init__("my_action", config)

        async def execute(self, match, context):
            return ActionResult(action_name=self.name, success=True)

    registry.register("my_action", MyAction)
    assert registry.get("my_action") is MyAction
    assert "my_action" in registry.list_actions()


def test_register_overwrites(registry):
    """Re-registering the same name should overwrite the previous class."""
    original = registry.get("log_analytics")
    assert original is LogAnalyticsAction

    class Replacement(BaseTriggerAction):
        def __init__(self, config=None):
            super().__init__("replacement", config)

        async def execute(self, match, context):
            return ActionResult(action_name=self.name, success=True)

    registry.register("log_analytics", Replacement)
    assert registry.get("log_analytics") is Replacement


def test_list_actions_sorted(registry):
    actions = registry.list_actions()
    assert actions == sorted(actions)


# ═══════════════════════════════════════════════════════════════════════════
# execute_for_match – basic
# ═══════════════════════════════════════════════════════════════════════════


@pytest.mark.asyncio
async def test_execute_for_match_none_configs(registry, rule_match, trigger_context):
    results = await registry.execute_for_match(rule_match, trigger_context, None)
    assert results == []


@pytest.mark.asyncio
async def test_execute_for_match_empty_configs(registry, rule_match, trigger_context):
    results = await registry.execute_for_match(rule_match, trigger_context, [])
    assert results == []


@pytest.mark.asyncio
async def test_execute_log_analytics(registry, rule_match, trigger_context):
    configs = [{"type": "log_analytics", "event_type": "test_event"}]
    results = await registry.execute_for_match(rule_match, trigger_context, configs)

    assert len(results) == 1
    assert results[0].success is True
    assert results[0].data["event_type"] == "test_event"
    assert results[0].data["rule_name"] == "test_rule"


@pytest.mark.asyncio
async def test_execute_unknown_action_type(registry, rule_match, trigger_context):
    configs = [{"type": "does_not_exist"}]
    results = await registry.execute_for_match(rule_match, trigger_context, configs)

    assert len(results) == 1
    assert results[0].success is False
    assert "unknown action type" in results[0].error


@pytest.mark.asyncio
async def test_execute_missing_type_key(registry, rule_match, trigger_context):
    configs = [{"event_type": "oops"}]  # no "type" key
    results = await registry.execute_for_match(rule_match, trigger_context, configs)

    assert len(results) == 1
    assert results[0].success is False
    assert "unknown action type" in results[0].error


@pytest.mark.asyncio
async def test_execute_multiple_actions(registry, rule_match, trigger_context):
    configs = [
        {"type": "log_analytics", "event_type": "event_a"},
        {"type": "log_analytics", "event_type": "event_b"},
    ]
    results = await registry.execute_for_match(rule_match, trigger_context, configs)

    assert len(results) == 2
    assert all(r.success for r in results)
    assert results[0].data["event_type"] == "event_a"
    assert results[1].data["event_type"] == "event_b"


@pytest.mark.asyncio
async def test_execute_error_isolation(registry, rule_match):
    """A failing action should not prevent subsequent actions from running."""
    ctx = TriggerContext(user_text="test")  # no knowledge_svc
    configs = [
        {"type": "enrich_prompt"},  # will fail (no knowledge_svc)
        {"type": "log_analytics"},  # should succeed
    ]
    results = await registry.execute_for_match(rule_match, ctx, configs)

    assert len(results) == 2
    assert results[0].success is False
    assert results[1].success is True


# ═══════════════════════════════════════════════════════════════════════════
# execute_for_match – retry logic
# ═══════════════════════════════════════════════════════════════════════════


@pytest.mark.asyncio
async def test_execute_no_retry_by_default(registry, rule_match):
    """max_retries=0 means no retries."""
    ctx = TriggerContext(user_text="test")
    configs = [{"type": "enrich_prompt"}]  # will fail
    results = await registry.execute_for_match(
        rule_match, ctx, configs, max_retries=0,
    )

    assert len(results) == 1
    assert results[0].success is False


@pytest.mark.asyncio
async def test_execute_with_retry_succeeds_eventually(registry, rule_match, trigger_context):
    """Action that fails once then succeeds should succeed with retry."""
    call_count = 0

    class FlakyAction(BaseTriggerAction):
        def __init__(self, config=None):
            super().__init__("flaky", config)

        async def execute(self, match, context):
            nonlocal call_count
            call_count += 1
            if call_count < 2:
                return ActionResult(action_name=self.name, success=False, error="transient")
            return ActionResult(action_name=self.name, success=True, data={"attempt": call_count})

    registry.register("flaky", FlakyAction)
    configs = [{"type": "flaky"}]
    results = await registry.execute_for_match(
        rule_match, trigger_context, configs, max_retries=2, retry_delay=0.01,
    )

    assert len(results) == 1
    assert results[0].success is True
    assert call_count == 2


@pytest.mark.asyncio
async def test_execute_retry_exhausted(registry, rule_match, trigger_context):
    """Action that always fails should exhaust retries and return failure."""

    class AlwaysFails(BaseTriggerAction):
        def __init__(self, config=None):
            super().__init__("always_fails", config)

        async def execute(self, match, context):
            return ActionResult(action_name=self.name, success=False, error="nope")

    registry.register("always_fails", AlwaysFails)
    configs = [{"type": "always_fails"}]
    results = await registry.execute_for_match(
        rule_match, trigger_context, configs, max_retries=1, retry_delay=0.01,
    )

    assert len(results) == 1
    assert results[0].success is False


@pytest.mark.asyncio
async def test_execute_retry_on_exception(registry, rule_match, trigger_context):
    """Actions that raise exceptions should also be retried."""
    call_count = 0

    class ExceptionAction(BaseTriggerAction):
        def __init__(self, config=None):
            super().__init__("exception_action", config)

        async def execute(self, match, context):
            nonlocal call_count
            call_count += 1
            if call_count < 2:
                raise RuntimeError("boom")
            return ActionResult(action_name=self.name, success=True)

    registry.register("exception_action", ExceptionAction)
    configs = [{"type": "exception_action"}]
    results = await registry.execute_for_match(
        rule_match, trigger_context, configs, max_retries=2, retry_delay=0.01,
    )

    assert len(results) == 1
    assert results[0].success is True
    assert call_count == 2


# ═══════════════════════════════════════════════════════════════════════════
# Chain building
# ═══════════════════════════════════════════════════════════════════════════


@pytest.mark.asyncio
async def test_execute_chain_config(registry, rule_match, trigger_context):
    configs = [{
        "type": "chain",
        "actions": [
            {"type": "log_analytics", "event_type": "step1"},
            {"type": "log_analytics", "event_type": "step2"},
        ],
        "stop_on_error": False,
    }]
    results = await registry.execute_for_match(rule_match, trigger_context, configs)

    assert len(results) == 1
    assert results[0].success is True
    assert results[0].data["executed"] == 2


@pytest.mark.asyncio
async def test_execute_chain_stop_on_error(registry, rule_match):
    ctx = TriggerContext(user_text="test")
    configs = [{
        "type": "chain",
        "actions": [
            {"type": "enrich_prompt"},  # will fail
            {"type": "log_analytics"},  # should not run
        ],
        "stop_on_error": True,
    }]
    results = await registry.execute_for_match(rule_match, ctx, configs)

    assert len(results) == 1
    assert results[0].success is False
    assert results[0].data["executed"] == 1


@pytest.mark.asyncio
async def test_execute_chain_empty_actions(registry, rule_match, trigger_context):
    configs = [{"type": "chain", "actions": []}]
    results = await registry.execute_for_match(rule_match, trigger_context, configs)

    assert len(results) == 1
    assert results[0].success is True
    assert results[0].data["executed"] == 0


@pytest.mark.asyncio
async def test_chain_skips_unknown_inner_actions(registry, rule_match, trigger_context):
    configs = [{
        "type": "chain",
        "actions": [
            {"type": "unknown_action"},
            {"type": "log_analytics"},
        ],
    }]
    results = await registry.execute_for_match(rule_match, trigger_context, configs)

    assert len(results) == 1
    # Only log_analytics should be in the chain (unknown skipped during build)
    assert results[0].data["executed"] == 1


# ═══════════════════════════════════════════════════════════════════════════
# Conditional building
# ═══════════════════════════════════════════════════════════════════════════


@pytest.mark.asyncio
async def test_execute_conditional_met(registry, rule_match, trigger_context):
    configs = [{
        "type": "conditional",
        "condition": "confidence >= 0.8",
        "action": {"type": "log_analytics", "event_type": "cond_event"},
    }]
    results = await registry.execute_for_match(rule_match, trigger_context, configs)

    assert len(results) == 1
    assert results[0].success is True
    assert results[0].data["met"] is True


@pytest.mark.asyncio
async def test_execute_conditional_not_met(registry, trigger_context):
    match = RuleMatch(rule_name="low", confidence=0.3, contexts=[])
    configs = [{
        "type": "conditional",
        "condition": "confidence >= 0.8",
        "action": {"type": "log_analytics"},
    }]
    results = await registry.execute_for_match(match, trigger_context, configs)

    assert len(results) == 1
    assert results[0].data["met"] is False
    assert results[0].data["skipped"] is True


@pytest.mark.asyncio
async def test_execute_conditional_always(registry, rule_match, trigger_context):
    configs = [{
        "type": "conditional",
        "condition": "always",
        "action": {"type": "log_analytics"},
    }]
    results = await registry.execute_for_match(rule_match, trigger_context, configs)

    assert len(results) == 1
    assert results[0].data["met"] is True


@pytest.mark.asyncio
async def test_execute_conditional_unknown_inner_falls_back(registry, rule_match, trigger_context):
    """When the inner action type is unknown, it falls back to log_analytics."""
    configs = [{
        "type": "conditional",
        "condition": "always",
        "action": {"type": "nonexistent"},
    }]
    results = await registry.execute_for_match(rule_match, trigger_context, configs)

    assert len(results) == 1
    assert results[0].success is True
    # The fallback action is a LogAnalyticsAction with event_type "fallback"
    assert results[0].data["met"] is True


@pytest.mark.asyncio
async def test_execute_conditional_no_condition_defaults_always(registry, rule_match, trigger_context):
    """Missing condition key should default to 'always'."""
    configs = [{
        "type": "conditional",
        "action": {"type": "log_analytics"},
    }]
    results = await registry.execute_for_match(rule_match, trigger_context, configs)

    assert len(results) == 1
    assert results[0].data["met"] is True


# ═══════════════════════════════════════════════════════════════════════════
# Singleton
# ═══════════════════════════════════════════════════════════════════════════


def test_singleton_returns_same_instance():
    reset_action_registry()
    r1 = get_action_registry()
    r2 = get_action_registry()
    assert r1 is r2
    reset_action_registry()


def test_reset_clears_singleton():
    reset_action_registry()
    r1 = get_action_registry()
    reset_action_registry()
    r2 = get_action_registry()
    assert r1 is not r2
    reset_action_registry()


# ═══════════════════════════════════════════════════════════════════════════
# Duration tracking
# ═══════════════════════════════════════════════════════════════════════════


@pytest.mark.asyncio
async def test_action_result_has_duration(registry, rule_match, trigger_context):
    configs = [{"type": "log_analytics"}]
    results = await registry.execute_for_match(rule_match, trigger_context, configs)

    assert len(results) == 1
    assert results[0].duration_ms >= 0
