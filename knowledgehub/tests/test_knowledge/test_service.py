"""Tests for the KnowledgeService orchestration layer.

Uses an in-memory SQLite DB and a fake vector store / embedder to
test the service logic without external dependencies.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from unittest.mock import AsyncMock, patch

import pytest
from sqlalchemy import select

from src.knowledge.service import (
    BulkDocument,
    KnowledgeSearchResult,
    KnowledgeService,
    _parse_json_array,
)
from src.knowledge.vectorstore import DocumentRecord, SearchResult, VectorStore
from src.knowledge.embeddings import EmbeddingProvider
from src.shared.models import (
    ContentType,
    Conversation,
    KnowledgeItem,
    Message,
    MessageRole,
)


# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------


class FakeEmbedder(EmbeddingProvider):
    """Returns deterministic fake embeddings of configurable dimension."""

    def __init__(self, dim: int = 384):
        self._dim = dim
        self.call_count = 0

    async def embed(self, texts: list[str]) -> list[list[float]]:
        self.call_count += 1
        return [[0.1] * self._dim for _ in texts]

    @property
    def dimension(self) -> int:
        return self._dim


class FakeVectorStore(VectorStore):
    """In-memory vector store for tests."""

    def __init__(self):
        self._store: dict[str, dict] = {}

    async def add(self, texts, metadatas, ids, embeddings=None) -> list[str]:
        for uid, text, meta in zip(ids, texts, metadatas):
            self._store[uid] = {"text": text, "metadata": meta, "embedding": None}
        return ids

    async def search(self, query_embedding, n_results=5, filter=None) -> list[SearchResult]:
        results = []
        for uid, data in list(self._store.items())[:n_results]:
            results.append(SearchResult(
                id=uid, content=data["text"], score=0.85, metadata=data["metadata"],
            ))
        return results

    async def delete(self, ids) -> bool:
        for uid in ids:
            self._store.pop(uid, None)
        return True

    async def get(self, ids) -> list[DocumentRecord]:
        return [
            DocumentRecord(id=uid, content=self._store[uid]["text"], metadata=self._store[uid]["metadata"])
            for uid in ids if uid in self._store
        ]

    async def update(self, id, text=None, metadata=None, embedding=None) -> bool:
        if id not in self._store:
            return False
        if text is not None:
            self._store[id]["text"] = text
        if metadata is not None:
            self._store[id]["metadata"].update(metadata)
        return True


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def fake_embedder():
    return FakeEmbedder()


@pytest.fixture
def fake_vectorstore():
    return FakeVectorStore()


@pytest.fixture
def service(db_session, fake_vectorstore, fake_embedder):
    return KnowledgeService(
        session=db_session,
        vectorstore=fake_vectorstore,
        embedder=fake_embedder,
    )


# ═══════════════════════════════════════════════════════════════════════════
# 1. add_knowledge
# ═══════════════════════════════════════════════════════════════════════════


@pytest.mark.asyncio
async def test_add_knowledge_creates_item(service, db_session):
    item = await service.add_knowledge(
        content="Python is a programming language",
        contexts=["tech"],
        source_type="manual",
    )
    assert item.id is not None
    assert item.content == "Python is a programming language"
    assert item.content_type == ContentType.MANUAL
    assert item.contexts == ["tech"]
    assert item.embedding_id is not None
    assert item.verified is False


@pytest.mark.asyncio
async def test_add_knowledge_deduplicates(service):
    item1 = await service.add_knowledge(
        content="Same content",
        contexts=["ctx"],
        source_type="manual",
    )
    item2 = await service.add_knowledge(
        content="Same content",
        contexts=["ctx"],
        source_type="manual",
    )
    assert item1.id == item2.id


@pytest.mark.asyncio
async def test_add_knowledge_with_source_message(service, db_session):
    conv = Conversation(session_id="sess-test", metadata_json={})
    db_session.add(conv)
    await db_session.flush()

    msg = Message(conversation_id=conv.id, role=MessageRole.USER, content="test")
    db_session.add(msg)
    await db_session.flush()

    item = await service.add_knowledge(
        content="Extracted fact",
        contexts=["project"],
        source_type="conversation_extract",
        source_message_id=msg.id,
    )
    assert item.source_message_id == msg.id
    assert item.content_type == ContentType.CONVERSATION_EXTRACT


@pytest.mark.asyncio
async def test_add_knowledge_stores_in_vectorstore(service, fake_vectorstore):
    await service.add_knowledge(
        content="Vector test",
        contexts=["test"],
        source_type="manual",
    )
    assert len(fake_vectorstore._store) == 1


# ═══════════════════════════════════════════════════════════════════════════
# 2. search_knowledge
# ═══════════════════════════════════════════════════════════════════════════


@pytest.mark.asyncio
async def test_search_knowledge_returns_enriched_results(service):
    item = await service.add_knowledge(
        content="FastAPI is a web framework",
        contexts=["tech", "python"],
        source_type="manual",
    )

    results = await service.search_knowledge(
        query="web framework",
        n_results=5,
        min_score=0.5,
    )
    assert len(results) >= 1
    assert isinstance(results[0], KnowledgeSearchResult)
    assert results[0].item.id == item.id
    assert results[0].score > 0


@pytest.mark.asyncio
async def test_search_knowledge_filters_by_context(service):
    await service.add_knowledge(
        content="Alpha project details",
        contexts=["alpha"],
        source_type="manual",
    )
    await service.add_knowledge(
        content="Beta project details",
        contexts=["beta"],
        source_type="manual",
    )

    results = await service.search_knowledge(
        query="project",
        contexts=["alpha"],
        n_results=5,
        min_score=0.5,
    )
    # Only alpha should match
    assert all("alpha" in r.item.contexts for r in results)


@pytest.mark.asyncio
async def test_search_knowledge_respects_min_score(service):
    await service.add_knowledge(
        content="Some content",
        contexts=["ctx"],
        source_type="manual",
    )
    # Our fake returns score=0.85, so min_score=0.9 should filter it out
    results = await service.search_knowledge(
        query="test",
        min_score=0.9,
    )
    assert results == []


@pytest.mark.asyncio
async def test_search_knowledge_empty_store(service):
    results = await service.search_knowledge(query="anything")
    assert results == []


# ═══════════════════════════════════════════════════════════════════════════
# 3. extract_knowledge_from_conversation
# ═══════════════════════════════════════════════════════════════════════════


@pytest.mark.asyncio
async def test_extract_knowledge_calls_llm(service, db_session):
    # Create a conversation with messages
    conv = Conversation(session_id="extract-sess", metadata_json={})
    db_session.add(conv)
    await db_session.flush()

    msg1 = Message(
        conversation_id=conv.id, role=MessageRole.USER,
        content="How do I deploy FastAPI?",
        detected_contexts=["fastapi"],
    )
    msg2 = Message(
        conversation_id=conv.id, role=MessageRole.ASSISTANT,
        content="Use uvicorn with --workers flag for production.",
    )
    db_session.add_all([msg1, msg2])
    await db_session.flush()

    # Mock the LLM to return extracted facts
    fake_llm = AsyncMock()
    fake_llm.chat = AsyncMock(return_value='["FastAPI can be deployed with uvicorn using --workers flag"]')

    with patch("src.knowledge.service.get_llm_provider", return_value=fake_llm):
        items = await service.extract_knowledge_from_conversation(conv.id)

    assert len(items) == 1
    assert "uvicorn" in items[0].content
    assert items[0].content_type == ContentType.CONVERSATION_EXTRACT
    assert items[0].verified is False

    # Messages should be marked as extracted
    result = await db_session.execute(
        select(Message).where(Message.conversation_id == conv.id)
    )
    for msg in result.scalars():
        assert msg.extracted_knowledge is True


@pytest.mark.asyncio
async def test_extract_knowledge_skips_already_extracted(service, db_session):
    conv = Conversation(session_id="extract-skip", metadata_json={})
    db_session.add(conv)
    await db_session.flush()

    msg = Message(
        conversation_id=conv.id, role=MessageRole.USER,
        content="Already processed", extracted_knowledge=True,
    )
    db_session.add(msg)
    await db_session.flush()

    items = await service.extract_knowledge_from_conversation(conv.id)
    assert items == []  # Nothing to extract


@pytest.mark.asyncio
async def test_extract_knowledge_force_reprocesses(service, db_session):
    conv = Conversation(session_id="extract-force", metadata_json={})
    db_session.add(conv)
    await db_session.flush()

    msg = Message(
        conversation_id=conv.id, role=MessageRole.USER,
        content="Re-extract me", extracted_knowledge=True,
    )
    db_session.add(msg)
    await db_session.flush()

    fake_llm = AsyncMock()
    fake_llm.chat = AsyncMock(return_value='["Re-extracted fact"]')

    with patch("src.knowledge.service.get_llm_provider", return_value=fake_llm):
        items = await service.extract_knowledge_from_conversation(conv.id, force=True)

    assert len(items) == 1


@pytest.mark.asyncio
async def test_extract_knowledge_empty_llm_response(service, db_session):
    conv = Conversation(session_id="extract-empty", metadata_json={})
    db_session.add(conv)
    await db_session.flush()

    msg = Message(
        conversation_id=conv.id, role=MessageRole.USER,
        content="Just small talk",
    )
    db_session.add(msg)
    await db_session.flush()

    fake_llm = AsyncMock()
    fake_llm.chat = AsyncMock(return_value="[]")

    with patch("src.knowledge.service.get_llm_provider", return_value=fake_llm):
        items = await service.extract_knowledge_from_conversation(conv.id)

    assert items == []


# ═══════════════════════════════════════════════════════════════════════════
# 4. build_rag_context
# ═══════════════════════════════════════════════════════════════════════════


@pytest.mark.asyncio
async def test_build_rag_context_returns_formatted_string(service):
    await service.add_knowledge(
        content="Important knowledge about deployment",
        contexts=["devops"],
        source_type="manual",
    )

    context = await service.build_rag_context(
        query="deployment",
        detected_contexts=["devops"],
    )
    assert "Important knowledge about deployment" in context
    assert "[Source:" in context


@pytest.mark.asyncio
async def test_build_rag_context_respects_max_tokens(service):
    # Add a very long knowledge item
    long_content = "A" * 10000
    await service.add_knowledge(
        content=long_content,
        contexts=["test"],
        source_type="manual",
    )

    context = await service.build_rag_context(
        query="test",
        detected_contexts=["test"],
        max_tokens=100,  # ~400 chars
    )
    # Should be truncated to roughly max_tokens*4 chars
    assert len(context) <= 500


@pytest.mark.asyncio
async def test_build_rag_context_empty_when_no_knowledge(service):
    context = await service.build_rag_context(
        query="nonexistent topic",
        detected_contexts=["missing"],
    )
    assert context == ""


# ═══════════════════════════════════════════════════════════════════════════
# 5. verify_knowledge
# ═══════════════════════════════════════════════════════════════════════════


@pytest.mark.asyncio
async def test_verify_knowledge_approve(service, fake_vectorstore):
    item = await service.add_knowledge(
        content="Verified fact",
        contexts=["qa"],
        source_type="manual",
    )
    assert item.verified is False

    updated = await service.verify_knowledge(item.id, verified=True, verified_by="admin")
    assert updated.verified is True
    assert updated.created_by == "admin"
    # Embedding should still exist in vector store
    assert item.embedding_id in fake_vectorstore._store


@pytest.mark.asyncio
async def test_verify_knowledge_reject_removes_from_vectorstore(service, fake_vectorstore):
    item = await service.add_knowledge(
        content="Rejected fact",
        contexts=["qa"],
        source_type="manual",
    )
    embedding_id = item.embedding_id
    assert embedding_id in fake_vectorstore._store

    updated = await service.verify_knowledge(item.id, verified=False, verified_by="admin")
    assert updated.verified is False
    assert updated.embedding_id is None
    assert embedding_id not in fake_vectorstore._store


@pytest.mark.asyncio
async def test_verify_knowledge_not_found(service):
    from src.shared.exceptions import NotFoundError
    with pytest.raises(NotFoundError):
        await service.verify_knowledge("nonexistent-id", verified=True, verified_by="admin")


# ═══════════════════════════════════════════════════════════════════════════
# 6. bulk_import
# ═══════════════════════════════════════════════════════════════════════════


@pytest.mark.asyncio
async def test_bulk_import_success(service, fake_vectorstore):
    docs = [
        BulkDocument(content="Document one content here", source="test"),
        BulkDocument(content="Document two content here", source="test"),
    ]
    result = await service.bulk_import(
        documents=docs,
        contexts=["import_test"],
        chunk_size=50,
        chunk_overlap=10,
    )
    assert result.total == 2
    assert result.imported == 2
    assert result.errors == []
    # Items should be in vector store
    assert len(fake_vectorstore._store) > 0


@pytest.mark.asyncio
async def test_bulk_import_empty_document(service):
    docs = [BulkDocument(content="", source="test")]
    result = await service.bulk_import(
        documents=docs,
        contexts=["test"],
    )
    assert result.total == 1
    assert result.imported == 0
    assert len(result.errors) == 1


@pytest.mark.asyncio
async def test_bulk_import_deduplication(service, fake_vectorstore):
    docs = [
        BulkDocument(content="Same short text", source="test"),
        BulkDocument(content="Same short text", source="test"),
    ]
    result = await service.bulk_import(
        documents=docs,
        contexts=["dedup"],
        chunk_size=1000,
    )
    assert result.imported == 2
    # Due to dedup in add_knowledge, only one vector store entry
    assert len(fake_vectorstore._store) == 1


# ═══════════════════════════════════════════════════════════════════════════
# _parse_json_array helper
# ═══════════════════════════════════════════════════════════════════════════


def test_parse_json_array_direct():
    assert _parse_json_array('["a", "b"]') == ["a", "b"]


def test_parse_json_array_from_code_block():
    text = '```json\n["fact 1", "fact 2"]\n```'
    assert _parse_json_array(text) == ["fact 1", "fact 2"]


def test_parse_json_array_embedded():
    text = 'Here are the facts: ["one", "two"] and that is all.'
    assert _parse_json_array(text) == ["one", "two"]


def test_parse_json_array_empty():
    assert _parse_json_array("[]") == []


def test_parse_json_array_invalid():
    assert _parse_json_array("not json at all") == []


def test_parse_json_array_filters_empty_strings():
    assert _parse_json_array('["good", "", "also good"]') == ["good", "also good"]
