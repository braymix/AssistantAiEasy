"""Tests for all SQLAlchemy models."""

import pytest
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

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


# ── Document ──────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_create_document(db_session: AsyncSession):
    doc = Document(title="Test Doc", content="Hello world", metadata_json={"source": "test"})
    db_session.add(doc)
    await db_session.flush()

    assert doc.id is not None
    assert len(doc.id) == 36

    result = await db_session.execute(select(Document).where(Document.id == doc.id))
    fetched = result.scalar_one()
    assert fetched.title == "Test Doc"
    assert fetched.content == "Hello world"
    assert fetched.metadata_json == {"source": "test"}


# ── Conversation & Messages ──────────────────────────────────────────────

@pytest.mark.asyncio
async def test_create_conversation_with_messages(db_session: AsyncSession):
    conv = Conversation(session_id="openwebui-session-123", metadata_json={"user": "alice"})
    db_session.add(conv)
    await db_session.flush()

    assert conv.id is not None
    assert len(conv.id) == 36

    # Add messages
    msg_user = Message(
        conversation_id=conv.id,
        role=MessageRole.USER,
        content="How do I configure the database?",
        detected_contexts=["database"],
    )
    msg_assistant = Message(
        conversation_id=conv.id,
        role=MessageRole.ASSISTANT,
        content="You can set DATABASE_URL in .env",
        extracted_knowledge=True,
    )
    db_session.add_all([msg_user, msg_assistant])
    await db_session.flush()

    assert msg_user.id != msg_assistant.id

    # Verify relationship
    result = await db_session.execute(
        select(Conversation).where(Conversation.id == conv.id)
    )
    fetched = result.scalar_one()
    assert len(fetched.messages) == 2
    assert fetched.messages[0].role == MessageRole.USER
    assert fetched.messages[1].extracted_knowledge is True


@pytest.mark.asyncio
async def test_message_roles(db_session: AsyncSession):
    conv = Conversation(session_id="sess-roles")
    db_session.add(conv)
    await db_session.flush()

    for role in MessageRole:
        msg = Message(conversation_id=conv.id, role=role, content=f"msg-{role.value}")
        db_session.add(msg)
    await db_session.flush()

    result = await db_session.execute(
        select(Message).where(Message.conversation_id == conv.id)
    )
    messages = list(result.scalars().all())
    assert len(messages) == 3
    roles = {m.role for m in messages}
    assert roles == {MessageRole.USER, MessageRole.ASSISTANT, MessageRole.SYSTEM}


# ── Context (hierarchy) ──────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_create_context_hierarchy(db_session: AsyncSession):
    parent = Context(name="engineering", description="Engineering department")
    db_session.add(parent)
    await db_session.flush()

    child = Context(
        name="backend",
        description="Backend team",
        parent_id=parent.id,
        metadata_json={"lead": "bob"},
    )
    db_session.add(child)
    await db_session.flush()

    # Fetch parent and check children
    result = await db_session.execute(select(Context).where(Context.id == parent.id))
    fetched_parent = result.scalar_one()
    assert len(fetched_parent.children) == 1
    assert fetched_parent.children[0].name == "backend"

    # Fetch child and check parent
    result = await db_session.execute(select(Context).where(Context.id == child.id))
    fetched_child = result.scalar_one()
    assert fetched_child.parent is not None
    assert fetched_child.parent.name == "engineering"


@pytest.mark.asyncio
async def test_context_unique_name(db_session: AsyncSession):
    db_session.add(Context(name="unique_ctx"))
    await db_session.flush()

    db_session.add(Context(name="unique_ctx"))
    with pytest.raises(Exception):  # IntegrityError
        await db_session.flush()


# ── KnowledgeItem ────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_create_knowledge_item_from_message(db_session: AsyncSession):
    conv = Conversation(session_id="sess-ki")
    db_session.add(conv)
    await db_session.flush()

    msg = Message(
        conversation_id=conv.id,
        role=MessageRole.ASSISTANT,
        content="The deploy process uses Docker Compose",
    )
    db_session.add(msg)
    await db_session.flush()

    ki = KnowledgeItem(
        source_message_id=msg.id,
        content="Deploy process uses Docker Compose",
        content_type=ContentType.CONVERSATION_EXTRACT,
        contexts=["deployment", "docker"],
        embedding_id="vec-abc-123",
        verified=False,
        created_by="auto-extract",
    )
    db_session.add(ki)
    await db_session.flush()

    assert ki.id is not None
    assert ki.verified is False

    # Check relationship from message side
    result = await db_session.execute(select(Message).where(Message.id == msg.id))
    fetched_msg = result.scalar_one()
    assert len(fetched_msg.knowledge_items) == 1
    assert fetched_msg.knowledge_items[0].content_type == ContentType.CONVERSATION_EXTRACT


@pytest.mark.asyncio
async def test_knowledge_item_without_message(db_session: AsyncSession):
    """KnowledgeItem can be created from upload / manual entry (no source_message)."""
    ki = KnowledgeItem(
        content="Manually entered knowledge",
        content_type=ContentType.MANUAL,
        contexts=["onboarding"],
        verified=True,
        created_by="admin",
    )
    db_session.add(ki)
    await db_session.flush()

    assert ki.source_message_id is None
    assert ki.verified is True


@pytest.mark.asyncio
async def test_knowledge_item_content_types(db_session: AsyncSession):
    for ct in ContentType:
        ki = KnowledgeItem(content=f"content-{ct.value}", content_type=ct)
        db_session.add(ki)
    await db_session.flush()

    result = await db_session.execute(select(KnowledgeItem))
    items = list(result.scalars().all())
    assert len(items) == 3
    types = {i.content_type for i in items}
    assert types == {ContentType.CONVERSATION_EXTRACT, ContentType.DOCUMENT, ContentType.MANUAL}


# ── DetectionRule ────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_create_detection_rule(db_session: AsyncSession):
    rule = DetectionRule(
        name="test_rule",
        description="A test detection rule",
        rule_type=RuleType.KEYWORD,
        rule_config={"keywords": ["test", "example"]},
        target_contexts=["testing"],
        priority=5,
        enabled=True,
    )
    db_session.add(rule)
    await db_session.flush()

    result = await db_session.execute(
        select(DetectionRule).where(DetectionRule.name == "test_rule")
    )
    fetched = result.scalar_one()
    assert fetched.rule_type == RuleType.KEYWORD
    assert fetched.rule_config["keywords"] == ["test", "example"]
    assert fetched.target_contexts == ["testing"]
    assert fetched.priority == 5
    assert fetched.enabled is True


@pytest.mark.asyncio
async def test_detection_rule_types(db_session: AsyncSession):
    for i, rt in enumerate(RuleType):
        rule = DetectionRule(
            name=f"rule_{rt.value}",
            rule_type=rt,
            rule_config={},
            priority=i,
        )
        db_session.add(rule)
    await db_session.flush()

    result = await db_session.execute(select(DetectionRule))
    rules = list(result.scalars().all())
    assert len(rules) == 4
    types = {r.rule_type for r in rules}
    assert types == {RuleType.KEYWORD, RuleType.REGEX, RuleType.SEMANTIC, RuleType.COMPOSITE}


@pytest.mark.asyncio
async def test_detection_rule_unique_name(db_session: AsyncSession):
    db_session.add(DetectionRule(name="dup_rule", rule_type=RuleType.KEYWORD))
    await db_session.flush()

    db_session.add(DetectionRule(name="dup_rule", rule_type=RuleType.REGEX))
    with pytest.raises(Exception):
        await db_session.flush()
