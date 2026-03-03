"""Tests for trigger actions and legacy trigger evaluation."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

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
    Trigger,
    TriggerAction,
    TriggerContext,
    _evaluate_condition,
    get_applicable_triggers,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


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
# Legacy trigger evaluation (backward compat)
# ═══════════════════════════════════════════════════════════════════════════


def test_high_confidence_triggers_all():
    triggers = get_applicable_triggers(confidence=0.9, topics=["general"])
    names = [t.name for t in triggers]
    assert "knowledge_enrichment" in names
    assert "high_confidence_alert" in names
    assert "analytics_logger" in names


def test_low_confidence_excludes_alert():
    triggers = get_applicable_triggers(confidence=0.2, topics=["general"])
    names = [t.name for t in triggers]
    assert "high_confidence_alert" not in names
    assert "analytics_logger" in names


def test_topic_filter():
    custom_triggers = [
        Trigger(name="specific", action=TriggerAction.ENRICH, min_confidence=0.0, target_topics=["security"]),
    ]
    result = get_applicable_triggers(confidence=0.5, topics=["general"], triggers=custom_triggers)
    assert len(result) == 0

    result = get_applicable_triggers(confidence=0.5, topics=["security"], triggers=custom_triggers)
    assert len(result) == 1


# ═══════════════════════════════════════════════════════════════════════════
# 1. SaveKnowledgeAction
# ═══════════════════════════════════════════════════════════════════════════


@pytest.mark.asyncio
async def test_save_knowledge_success(rule_match, trigger_context):
    mock_svc = AsyncMock()
    mock_item = MagicMock()
    mock_item.id = "item-1"
    mock_svc.add_knowledge = AsyncMock(return_value=mock_item)
    mock_svc.verify_knowledge = AsyncMock()
    trigger_context.knowledge_svc = mock_svc

    action = SaveKnowledgeAction(config={"auto_verify": False, "min_confidence": 0.5})
    result = await action.execute(rule_match, trigger_context)

    assert result.success is True
    assert result.data["item_id"] == "item-1"
    mock_svc.add_knowledge.assert_awaited_once()


@pytest.mark.asyncio
async def test_save_knowledge_auto_verify(rule_match, trigger_context):
    mock_svc = AsyncMock()
    mock_item = MagicMock()
    mock_item.id = "item-2"
    mock_svc.add_knowledge = AsyncMock(return_value=mock_item)
    mock_svc.verify_knowledge = AsyncMock()
    trigger_context.knowledge_svc = mock_svc

    action = SaveKnowledgeAction(config={"auto_verify": True})
    result = await action.execute(rule_match, trigger_context)

    assert result.success is True
    assert result.data["auto_verified"] is True
    mock_svc.verify_knowledge.assert_awaited_once()


@pytest.mark.asyncio
async def test_save_knowledge_below_confidence(trigger_context):
    match = RuleMatch(rule_name="low", confidence=0.2, contexts=[])
    action = SaveKnowledgeAction(config={"min_confidence": 0.5})
    result = await action.execute(match, trigger_context)

    assert result.success is True
    assert result.data.get("skipped") is True


@pytest.mark.asyncio
async def test_save_knowledge_no_service(rule_match):
    ctx = TriggerContext(user_text="test")
    action = SaveKnowledgeAction()
    result = await action.execute(rule_match, ctx)

    assert result.success is False
    assert "not available" in result.error


# ═══════════════════════════════════════════════════════════════════════════
# 2. NotifyAdminAction
# ═══════════════════════════════════════════════════════════════════════════


@pytest.mark.asyncio
async def test_notify_admin_log_channel(rule_match, trigger_context):
    action = NotifyAdminAction(config={"channels": ["log"]})
    result = await action.execute(rule_match, trigger_context)

    assert result.success is True
    assert "log" in result.data["notified"]


@pytest.mark.asyncio
async def test_notify_admin_custom_template(rule_match, trigger_context):
    action = NotifyAdminAction(config={
        "channels": ["log"],
        "template": "Alert: {rule_name} ({confidence:.1f})",
    })
    result = await action.execute(rule_match, trigger_context)

    assert "test_rule" in result.data["message"]
    assert "0.8" in result.data["message"]


@pytest.mark.asyncio
async def test_notify_admin_multiple_channels(rule_match, trigger_context):
    action = NotifyAdminAction(config={"channels": ["log", "email", "webhook"]})
    result = await action.execute(rule_match, trigger_context)

    assert "log" in result.data["notified"]
    assert "email" in result.data["notified"]
    assert "webhook" in result.data["notified"]


@pytest.mark.asyncio
async def test_notify_admin_unknown_channel(rule_match, trigger_context):
    action = NotifyAdminAction(config={"channels": ["sms"]})
    result = await action.execute(rule_match, trigger_context)

    assert result.success is False
    assert "unknown channel" in result.error


# ═══════════════════════════════════════════════════════════════════════════
# 3. EnrichPromptAction
# ═══════════════════════════════════════════════════════════════════════════


@pytest.mark.asyncio
async def test_enrich_prompt_success(rule_match, trigger_context):
    mock_svc = AsyncMock()
    mock_svc.build_rag_context = AsyncMock(return_value="RAG content here")
    trigger_context.knowledge_svc = mock_svc

    action = EnrichPromptAction(config={"max_tokens": 1500})
    result = await action.execute(rule_match, trigger_context)

    assert result.success is True
    assert result.data["rag_context"] == "RAG content here"
    assert result.data["max_tokens"] == 1500


@pytest.mark.asyncio
async def test_enrich_prompt_no_service(rule_match):
    ctx = TriggerContext(user_text="test")
    action = EnrichPromptAction()
    result = await action.execute(rule_match, ctx)

    assert result.success is False
    assert "not available" in result.error


@pytest.mark.asyncio
async def test_enrich_prompt_empty_rag(rule_match, trigger_context):
    mock_svc = AsyncMock()
    mock_svc.build_rag_context = AsyncMock(return_value="")
    trigger_context.knowledge_svc = mock_svc

    action = EnrichPromptAction()
    result = await action.execute(rule_match, trigger_context)

    assert result.success is True
    assert result.data["has_context"] is False


# ═══════════════════════════════════════════════════════════════════════════
# 4. TagConversationAction
# ═══════════════════════════════════════════════════════════════════════════


@pytest.mark.asyncio
async def test_tag_conversation_success(rule_match, db_session):
    from src.shared.models import Conversation
    conv = Conversation(session_id="tag-test", metadata_json={})
    db_session.add(conv)
    await db_session.flush()

    ctx = TriggerContext(
        session=db_session,
        conversation_id=conv.id,
        user_text="test",
    )
    action = TagConversationAction(config={"tags": ["alpha", "beta"]})
    result = await action.execute(rule_match, ctx)

    assert result.success is True
    assert "alpha" in result.data["tags"]
    assert "beta" in result.data["tags"]


@pytest.mark.asyncio
async def test_tag_conversation_no_conv_id(rule_match):
    ctx = TriggerContext(user_text="test", session=MagicMock())
    action = TagConversationAction(config={"tags": ["x"]})
    result = await action.execute(rule_match, ctx)

    assert result.success is False
    assert "no conversation_id" in result.error


@pytest.mark.asyncio
async def test_tag_conversation_falls_back_to_contexts(db_session):
    from src.shared.models import Conversation
    conv = Conversation(session_id="tag-fallback", metadata_json={})
    db_session.add(conv)
    await db_session.flush()

    match = RuleMatch(rule_name="r", confidence=0.8, contexts=["ctx_a"])
    ctx = TriggerContext(session=db_session, conversation_id=conv.id, user_text="test")

    action = TagConversationAction(config={})  # no tags specified
    result = await action.execute(match, ctx)

    assert result.success is True
    assert "ctx_a" in result.data["tags"]


# ═══════════════════════════════════════════════════════════════════════════
# 5. LogAnalyticsAction
# ═══════════════════════════════════════════════════════════════════════════


@pytest.mark.asyncio
async def test_log_analytics_default(rule_match, trigger_context):
    action = LogAnalyticsAction()
    result = await action.execute(rule_match, trigger_context)

    assert result.success is True
    assert result.data["event_type"] == "context_detected"
    assert result.data["rule_name"] == "test_rule"
    assert result.data["confidence"] == 0.85


@pytest.mark.asyncio
async def test_log_analytics_custom_event(rule_match, trigger_context):
    action = LogAnalyticsAction(config={
        "event_type": "important_match",
        "extra_metadata": {"source": "unit_test"},
    })
    result = await action.execute(rule_match, trigger_context)

    assert result.data["event_type"] == "important_match"
    assert result.data["source"] == "unit_test"


# ═══════════════════════════════════════════════════════════════════════════
# 6. ChainAction
# ═══════════════════════════════════════════════════════════════════════════


@pytest.mark.asyncio
async def test_chain_action_all_succeed(rule_match, trigger_context):
    a1 = LogAnalyticsAction(config={"event_type": "step_1"})
    a2 = LogAnalyticsAction(config={"event_type": "step_2"})
    chain = ChainAction(actions=[a1, a2])

    result = await chain.execute(rule_match, trigger_context)

    assert result.success is True
    assert result.data["executed"] == 2
    assert all(r["success"] for r in result.data["chain_results"])


@pytest.mark.asyncio
async def test_chain_action_continues_on_error(rule_match):
    ctx = TriggerContext(user_text="test")
    a1 = EnrichPromptAction()  # will fail (no knowledge_svc)
    a2 = LogAnalyticsAction()  # will succeed

    chain = ChainAction(actions=[a1, a2], config={"stop_on_error": False})
    result = await chain.execute(rule_match, ctx)

    assert result.success is False  # overall: one failed
    assert result.data["executed"] == 2  # but both ran


@pytest.mark.asyncio
async def test_chain_action_stops_on_error(rule_match):
    ctx = TriggerContext(user_text="test")
    a1 = EnrichPromptAction()  # will fail
    a2 = LogAnalyticsAction()  # should not run

    chain = ChainAction(actions=[a1, a2], config={"stop_on_error": True})
    result = await chain.execute(rule_match, ctx)

    assert result.success is False
    assert result.data["executed"] == 1  # stopped after first


@pytest.mark.asyncio
async def test_chain_action_empty(rule_match, trigger_context):
    chain = ChainAction(actions=[])
    result = await chain.execute(rule_match, trigger_context)

    assert result.success is True
    assert result.data["executed"] == 0


# ═══════════════════════════════════════════════════════════════════════════
# 7. ConditionalAction
# ═══════════════════════════════════════════════════════════════════════════


@pytest.mark.asyncio
async def test_conditional_always(rule_match, trigger_context):
    inner = LogAnalyticsAction()
    cond = ConditionalAction(action=inner, config={"condition": "always"})
    result = await cond.execute(rule_match, trigger_context)

    assert result.success is True
    assert result.data["met"] is True


@pytest.mark.asyncio
async def test_conditional_confidence_met(rule_match, trigger_context):
    inner = LogAnalyticsAction()
    cond = ConditionalAction(action=inner, config={"condition": "confidence >= 0.8"})
    result = await cond.execute(rule_match, trigger_context)

    assert result.data["met"] is True


@pytest.mark.asyncio
async def test_conditional_confidence_not_met(trigger_context):
    match = RuleMatch(rule_name="low", confidence=0.3, contexts=[])
    inner = LogAnalyticsAction()
    cond = ConditionalAction(action=inner, config={"condition": "confidence >= 0.8"})
    result = await cond.execute(match, trigger_context)

    assert result.data["met"] is False
    assert result.data["skipped"] is True


@pytest.mark.asyncio
async def test_conditional_context_contains(rule_match, trigger_context):
    inner = LogAnalyticsAction()
    cond = ConditionalAction(action=inner, config={"condition": "context_contains database"})
    result = await cond.execute(rule_match, trigger_context)

    assert result.data["met"] is True


@pytest.mark.asyncio
async def test_conditional_context_not_contains(rule_match, trigger_context):
    inner = LogAnalyticsAction()
    cond = ConditionalAction(action=inner, config={"condition": "context_contains security"})
    result = await cond.execute(rule_match, trigger_context)

    assert result.data["met"] is False


@pytest.mark.asyncio
async def test_conditional_has_conversation(rule_match, trigger_context):
    inner = LogAnalyticsAction()
    cond = ConditionalAction(action=inner, config={"condition": "has_conversation"})
    result = await cond.execute(rule_match, trigger_context)

    assert result.data["met"] is True  # trigger_context has conversation_id


@pytest.mark.asyncio
async def test_conditional_has_conversation_missing(rule_match):
    ctx = TriggerContext(user_text="test")  # no conversation_id
    inner = LogAnalyticsAction()
    cond = ConditionalAction(action=inner, config={"condition": "has_conversation"})
    result = await cond.execute(rule_match, ctx)

    assert result.data["met"] is False


# ═══════════════════════════════════════════════════════════════════════════
# _evaluate_condition helper
# ═══════════════════════════════════════════════════════════════════════════


def test_condition_always():
    match = RuleMatch(rule_name="r", confidence=0.5, contexts=[])
    ctx = TriggerContext()
    assert _evaluate_condition("always", match, ctx) is True


def test_condition_confidence_operators():
    match = RuleMatch(rule_name="r", confidence=0.7, contexts=[])
    ctx = TriggerContext()
    assert _evaluate_condition("confidence >= 0.5", match, ctx) is True
    assert _evaluate_condition("confidence >= 0.9", match, ctx) is False
    assert _evaluate_condition("confidence > 0.5", match, ctx) is True
    assert _evaluate_condition("confidence < 0.8", match, ctx) is True
    assert _evaluate_condition("confidence <= 0.7", match, ctx) is True


def test_condition_context_contains():
    match = RuleMatch(rule_name="r", confidence=0.5, contexts=["alpha", "beta"])
    ctx = TriggerContext()
    assert _evaluate_condition("context_contains alpha", match, ctx) is True
    assert _evaluate_condition("context_contains gamma", match, ctx) is False


def test_condition_unknown():
    match = RuleMatch(rule_name="r", confidence=0.5, contexts=[])
    ctx = TriggerContext()
    assert _evaluate_condition("gibberish", match, ctx) is False


# ═══════════════════════════════════════════════════════════════════════════
# Dataclass defaults
# ═══════════════════════════════════════════════════════════════════════════


def test_action_result_defaults():
    r = ActionResult(action_name="test", success=True)
    assert r.data == {}
    assert r.error is None
    assert r.duration_ms == 0
    assert r.timestamp  # non-empty


def test_trigger_context_defaults():
    ctx = TriggerContext()
    assert ctx.session is None
    assert ctx.knowledge_svc is None
    assert ctx.conversation_id is None
    assert ctx.user_text == ""
    assert ctx.messages == []
    assert ctx.metadata == {}
