"""Integration tests for full KnowledgeHub flow.

Tests end-to-end flows combining detection, knowledge, and conversation.
Uses in-memory SQLite and mock LLM/vectorstore — no external services.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest
from sqlalchemy import select

from src.detection.engine import DetectionEngine
from src.detection.rules import DetectionContext, KeywordRule
from src.knowledge.service import KnowledgeService
from src.shared.models import (
    ContentType,
    Context,
    Conversation,
    DetectionRule,
    KnowledgeItem,
    Message,
    MessageRole,
    RuleType,
)
from tests.conftest import FakeEmbedder, FakeVectorStore


# ── Fixtures ───────────────────────────────────────────────────────────────


@pytest.fixture
def vectorstore():
    return FakeVectorStore()


@pytest.fixture
def embedder():
    return FakeEmbedder()


@pytest.fixture
def knowledge_svc(db_session, vectorstore, embedder):
    return KnowledgeService(
        session=db_session,
        vectorstore=vectorstore,
        embedder=embedder,
    )


@pytest.fixture
async def populated_db(db_session):
    """Set up contexts, rules, and sample knowledge."""
    # Contexts
    ctx_db = Context(name="database", description="Database topics")
    ctx_proc = Context(name="procedures", description="Procedure operative")
    db_session.add_all([ctx_db, ctx_proc])
    await db_session.flush()

    # Rules
    rules = [
        DetectionRule(
            name="database_kw",
            rule_type=RuleType.KEYWORD,
            rule_config={"keywords": ["database", "sql", "query"]},
            target_contexts=["database"],
            priority=10,
            enabled=True,
        ),
        DetectionRule(
            name="procedure_regex",
            rule_type=RuleType.REGEX,
            rule_config={"patterns": [r"come (si fa|faccio|posso)", r"procedura per"]},
            target_contexts=["procedures"],
            priority=10,
            enabled=True,
        ),
    ]
    for rule in rules:
        db_session.add(rule)
    await db_session.flush()

    return {"contexts": [ctx_db, ctx_proc], "rules": rules}


# ═══════════════════════════════════════════════════════════════════════════════
# 1. Conversation with knowledge extraction
# ═══════════════════════════════════════════════════════════════════════════════


@pytest.mark.asyncio
async def test_conversation_with_knowledge_extraction(
    db_session, knowledge_svc, populated_db,
):
    """Full flow: create conversation → detect contexts → extract knowledge."""
    # 1. Create conversation
    conv = Conversation(session_id="integ-test-001", metadata_json={})
    db_session.add(conv)
    await db_session.flush()

    # 2. Add user message
    user_msg = Message(
        conversation_id=conv.id,
        role=MessageRole.USER,
        content="Come si configura il database con connection pooling?",
    )
    db_session.add(user_msg)
    await db_session.flush()

    # 3. Detect context
    engine = DetectionEngine(db_session)
    result = await engine.detect(user_msg.content)
    assert result.confidence > 0
    assert "database" in result.suggested_topics

    # 4. Add assistant response
    assistant_msg = Message(
        conversation_id=conv.id,
        role=MessageRole.ASSISTANT,
        content="Per configurare il connection pooling, imposta pool_size=20 nel database URL.",
    )
    db_session.add(assistant_msg)
    await db_session.flush()

    # 5. Extract knowledge from conversation
    fake_llm = AsyncMock()
    fake_llm.chat = AsyncMock(
        return_value='["Il connection pooling si configura con pool_size=20 nel database URL"]'
    )

    with patch("src.knowledge.service.get_llm_provider", return_value=fake_llm):
        items = await knowledge_svc.extract_knowledge_from_conversation(conv.id)

    assert len(items) >= 1
    assert items[0].content_type == ContentType.CONVERSATION_EXTRACT
    assert items[0].verified is False

    # 6. Verify the knowledge item
    verified = await knowledge_svc.verify_knowledge(items[0].id, verified=True, verified_by="admin")
    assert verified.verified is True


# ═══════════════════════════════════════════════════════════════════════════════
# 2. RAG enrichment flow
# ═══════════════════════════════════════════════════════════════════════════════


@pytest.mark.asyncio
async def test_rag_enrichment_flow(
    db_session, knowledge_svc, populated_db,
):
    """Full flow: add knowledge → detect context → enrich with RAG."""
    # 1. Seed knowledge
    await knowledge_svc.add_knowledge(
        content="Il database PostgreSQL usa la porta 5432 e supporta connection pooling via pgbouncer.",
        contexts=["database"],
        source_type="manual",
    )

    # 2. Detect context on user query
    engine = DetectionEngine(db_session)
    user_text = "Come faccio a configurare il database?"
    detect_result = await engine.detect(user_text)
    assert detect_result.confidence > 0

    # 3. Build RAG context
    rag_context = await knowledge_svc.build_rag_context(
        query=user_text,
        detected_contexts=detect_result.suggested_topics,
    )
    assert "PostgreSQL" in rag_context or "database" in rag_context.lower()

    # 4. Full detect_and_enrich
    messages = [{"role": "user", "content": user_text}]
    enriched = await engine.detect_and_enrich(
        messages=messages,
        knowledge_svc=knowledge_svc,
    )
    assert enriched.detected.confidence > 0
    # Enriched messages should contain RAG system message if knowledge was found
    if enriched.rag_context:
        assert len(enriched.enriched_messages) > len(messages)


# ═══════════════════════════════════════════════════════════════════════════════
# 3. Admin rule change affects detection
# ═══════════════════════════════════════════════════════════════════════════════


@pytest.mark.asyncio
async def test_admin_rule_change_affects_detection(db_session, populated_db):
    """Adding a new rule changes detection results."""
    engine = DetectionEngine(db_session)

    # Initially "kubernetes" should not be detected
    result1 = await engine.detect("deploy to kubernetes cluster")
    k8s_topics_before = [t for t in result1.suggested_topics if "k8s" in t or "kubernetes" in t]
    assert len(k8s_topics_before) == 0

    # Add a new rule for kubernetes
    new_rule = DetectionRule(
        name="kubernetes_deploy",
        rule_type=RuleType.KEYWORD,
        rule_config={"keywords": ["kubernetes", "k8s", "kubectl"]},
        target_contexts=["kubernetes"],
        priority=10,
        enabled=True,
    )
    db_session.add(new_rule)
    await db_session.flush()

    # Reload rules (simulates admin hot-reload)
    engine2 = DetectionEngine(db_session)
    result2 = await engine2.detect("deploy to kubernetes cluster")
    assert "kubernetes" in result2.suggested_topics


# ═══════════════════════════════════════════════════════════════════════════════
# 4. Knowledge verification flow
# ═══════════════════════════════════════════════════════════════════════════════


@pytest.mark.asyncio
async def test_knowledge_verification_flow(
    db_session, knowledge_svc, vectorstore,
):
    """Full verification lifecycle: add → verify → reject."""
    # 1. Add unverified knowledge
    item = await knowledge_svc.add_knowledge(
        content="Fact to be reviewed by admin",
        contexts=["general"],
        source_type="conversation_extract",
    )
    assert item.verified is False
    assert item.embedding_id is not None
    original_embedding_id = item.embedding_id

    # 2. Approve the knowledge
    approved = await knowledge_svc.verify_knowledge(item.id, verified=True, verified_by="reviewer1")
    assert approved.verified is True
    assert approved.embedding_id is not None  # Still in vectorstore

    # 3. Search should find it
    results = await knowledge_svc.search_knowledge(query="fact reviewed", min_score=0.5)
    assert any(r.item.id == item.id for r in results)

    # 4. Now reject it (admin changes mind)
    rejected = await knowledge_svc.verify_knowledge(item.id, verified=False, verified_by="reviewer2")
    assert rejected.verified is False
    assert rejected.embedding_id is None  # Removed from vectorstore
    assert original_embedding_id not in vectorstore._store
