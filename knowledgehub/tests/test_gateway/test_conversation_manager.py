"""Tests for ConversationManager service."""

import pytest
from sqlalchemy import select

from src.gateway.services.conversation import ConversationManager
from src.shared.models import (
    ContentType,
    Conversation,
    KnowledgeItem,
    Message,
    MessageRole,
)


# ── 1. create_or_get_conversation ──────────────────────────────────────────


@pytest.mark.asyncio
async def test_create_conversation_new(db_session):
    """First call with a session_id should create a new Conversation."""
    mgr = ConversationManager(db_session)
    conv = await mgr.create_or_get_conversation("sess-001")

    assert conv.id is not None
    assert conv.session_id == "sess-001"


@pytest.mark.asyncio
async def test_create_conversation_reuses_existing(db_session):
    """Subsequent calls with the same session_id should return the same Conversation."""
    mgr = ConversationManager(db_session)
    conv1 = await mgr.create_or_get_conversation("sess-002")
    conv2 = await mgr.create_or_get_conversation("sess-002")

    assert conv1.id == conv2.id


@pytest.mark.asyncio
async def test_create_conversation_different_sessions(db_session):
    """Different session_ids should produce distinct Conversations."""
    mgr = ConversationManager(db_session)
    conv_a = await mgr.create_or_get_conversation("sess-A")
    conv_b = await mgr.create_or_get_conversation("sess-B")

    assert conv_a.id != conv_b.id


# ── 2. add_message ─────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_add_message_user(db_session):
    mgr = ConversationManager(db_session)
    conv = await mgr.create_or_get_conversation("sess-msg")

    msg = await mgr.add_message(conv.id, "user", "Hello!")
    assert msg.role == MessageRole.USER
    assert msg.content == "Hello!"
    assert msg.conversation_id == conv.id


@pytest.mark.asyncio
async def test_add_message_assistant(db_session):
    mgr = ConversationManager(db_session)
    conv = await mgr.create_or_get_conversation("sess-msg-asst")

    msg = await mgr.add_message(conv.id, "assistant", "Hi there!")
    assert msg.role == MessageRole.ASSISTANT


@pytest.mark.asyncio
async def test_add_message_with_metadata(db_session):
    mgr = ConversationManager(db_session)
    conv = await mgr.create_or_get_conversation("sess-meta")

    msg = await mgr.add_message(
        conv.id, "user", "Tell me about Alpha",
        metadata={"detected_contexts": ["progetto_alpha"]},
    )
    assert msg.detected_contexts == ["progetto_alpha"]


@pytest.mark.asyncio
async def test_add_message_invalid_role_defaults_to_user(db_session):
    mgr = ConversationManager(db_session)
    conv = await mgr.create_or_get_conversation("sess-role")

    msg = await mgr.add_message(conv.id, "unknown_role", "test")
    assert msg.role == MessageRole.USER


# ── 3. get_conversation_history ────────────────────────────────────────────


@pytest.mark.asyncio
async def test_get_conversation_history(db_session):
    mgr = ConversationManager(db_session)
    conv = await mgr.create_or_get_conversation("sess-hist")

    await mgr.add_message(conv.id, "user", "First")
    await mgr.add_message(conv.id, "assistant", "Second")
    await mgr.add_message(conv.id, "user", "Third")

    history = await mgr.get_conversation_history(conv.id)
    assert len(history) == 3
    assert history[0].content == "First"
    assert history[2].content == "Third"


@pytest.mark.asyncio
async def test_get_conversation_history_limit(db_session):
    mgr = ConversationManager(db_session)
    conv = await mgr.create_or_get_conversation("sess-lim")

    for i in range(5):
        await mgr.add_message(conv.id, "user", f"msg-{i}")

    history = await mgr.get_conversation_history(conv.id, limit=3)
    assert len(history) == 3


@pytest.mark.asyncio
async def test_get_conversation_history_empty(db_session):
    mgr = ConversationManager(db_session)
    conv = await mgr.create_or_get_conversation("sess-empty")

    history = await mgr.get_conversation_history(conv.id)
    assert history == []


# ── 4. get_conversations_by_context ────────────────────────────────────────


@pytest.mark.asyncio
async def test_get_conversations_by_context(db_session):
    mgr = ConversationManager(db_session)

    # Create two conversations; only the first has matching context
    conv1 = await mgr.create_or_get_conversation("sess-ctx-1")
    await mgr.add_message(
        conv1.id, "user", "About alpha",
        metadata={"detected_contexts": ["progetto_alpha"]},
    )

    conv2 = await mgr.create_or_get_conversation("sess-ctx-2")
    await mgr.add_message(conv2.id, "user", "No context here")

    await db_session.flush()

    results = await mgr.get_conversations_by_context("progetto_alpha")
    result_ids = [c.id for c in results]
    assert conv1.id in result_ids
    assert conv2.id not in result_ids


@pytest.mark.asyncio
async def test_get_conversations_by_context_no_match(db_session):
    mgr = ConversationManager(db_session)
    results = await mgr.get_conversations_by_context("nonexistent_context")
    assert results == []


# ── 5. mark_knowledge_extracted ────────────────────────────────────────────


@pytest.mark.asyncio
async def test_mark_knowledge_extracted(db_session):
    mgr = ConversationManager(db_session)
    conv = await mgr.create_or_get_conversation("sess-ke")

    msg = await mgr.add_message(conv.id, "user", "Some knowledge here")
    assert msg.extracted_knowledge is False

    # Create a KnowledgeItem
    ki = KnowledgeItem(
        content="Extracted piece",
        content_type=ContentType.CONVERSATION_EXTRACT,
    )
    db_session.add(ki)
    await db_session.flush()

    await mgr.mark_knowledge_extracted(msg.id, ki.id)

    # Reload to verify
    result = await db_session.execute(select(Message).where(Message.id == msg.id))
    refreshed_msg = result.scalar_one()
    assert refreshed_msg.extracted_knowledge is True

    result_ki = await db_session.execute(select(KnowledgeItem).where(KnowledgeItem.id == ki.id))
    refreshed_ki = result_ki.scalar_one()
    assert refreshed_ki.source_message_id == msg.id


@pytest.mark.asyncio
async def test_mark_knowledge_extracted_missing_message(db_session):
    """Should not raise if the message doesn't exist."""
    mgr = ConversationManager(db_session)

    ki = KnowledgeItem(
        content="orphan",
        content_type=ContentType.MANUAL,
    )
    db_session.add(ki)
    await db_session.flush()

    # Should not raise
    await mgr.mark_knowledge_extracted("nonexistent-id", ki.id)


@pytest.mark.asyncio
async def test_mark_knowledge_extracted_missing_knowledge_item(db_session):
    """Should still mark the message even if the knowledge item doesn't exist."""
    mgr = ConversationManager(db_session)
    conv = await mgr.create_or_get_conversation("sess-ke-miss")
    msg = await mgr.add_message(conv.id, "user", "data")

    await mgr.mark_knowledge_extracted(msg.id, "nonexistent-ki-id")

    result = await db_session.execute(select(Message).where(Message.id == msg.id))
    refreshed = result.scalar_one()
    assert refreshed.extracted_knowledge is True
