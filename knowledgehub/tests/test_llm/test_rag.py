"""Tests for the RAG Orchestrator."""

from dataclasses import dataclass, field
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.llm.models import (
    ChatCompletion,
    ChatCompletionChunk,
    ChatMessage,
    Choice,
    ChoiceMessage,
    DeltaMessage,
    StreamChoice,
    UsageInfo,
)
from src.llm.rag import RAGOrchestrator, RAGResult, _QueryCache


# ═══════════════════════════════════════════════════════════════════════════════
# Mock helpers
# ═══════════════════════════════════════════════════════════════════════════════


@dataclass
class FakeKnowledgeItem:
    """Lightweight stand-in for the real KnowledgeItem ORM model."""
    content: str = "Test knowledge content"
    content_type: MagicMock = None
    contexts: list = field(default_factory=lambda: ["IT"])
    id: str = "ki-1"

    def __post_init__(self):
        if self.content_type is None:
            self.content_type = MagicMock(value="document")


@dataclass
class FakeSearchResult:
    """Lightweight stand-in for KnowledgeSearchResult."""
    item: FakeKnowledgeItem = None
    score: float = 0.85
    highlights: list = field(default_factory=list)

    def __post_init__(self):
        if self.item is None:
            self.item = FakeKnowledgeItem()


@pytest.fixture
def mock_llm() -> AsyncMock:
    llm = AsyncMock()
    llm.complete = AsyncMock(return_value=ChatCompletion(
        id="test-id",
        model="test-model",
        choices=[Choice(message=ChoiceMessage(content="LLM response"))],
        usage=UsageInfo(prompt_tokens=10, completion_tokens=20),
    ))
    llm.chat = AsyncMock(return_value="LLM chat response")
    return llm


@pytest.fixture
def mock_knowledge() -> AsyncMock:
    svc = AsyncMock()
    svc.search_knowledge = AsyncMock(return_value=[
        FakeSearchResult(
            item=FakeKnowledgeItem(content="Database backups run at 2am"),
            score=0.9,
        ),
        FakeSearchResult(
            item=FakeKnowledgeItem(content="Max upload size is 50MB"),
            score=0.7,
        ),
    ])
    svc.add_knowledge = AsyncMock(return_value=FakeKnowledgeItem())
    return svc


@pytest.fixture
def mock_settings() -> MagicMock:
    return MagicMock()


@pytest.fixture
def rag(mock_llm, mock_knowledge, mock_settings) -> RAGOrchestrator:
    return RAGOrchestrator(
        llm_provider=mock_llm,
        knowledge_service=mock_knowledge,
        settings=mock_settings,
        show_sources=True,
        enable_rerank=False,
        cache_ttl=300.0,
    )


# ═══════════════════════════════════════════════════════════════════════════════
# QueryCache
# ═══════════════════════════════════════════════════════════════════════════════


class TestQueryCache:
    def test_put_and_get(self) -> None:
        cache = _QueryCache(maxsize=10, ttl=60.0)
        cache.put("k1", "v1")
        assert cache.get("k1") == "v1"

    def test_ttl_expiry(self) -> None:
        cache = _QueryCache(ttl=0.0)
        cache.put("k1", "v1")
        assert cache.get("k1") is None

    def test_maxsize_eviction(self) -> None:
        cache = _QueryCache(maxsize=2, ttl=300.0)
        cache.put("k1", "v1")
        cache.put("k2", "v2")
        cache.put("k3", "v3")
        assert cache.get("k1") is None
        assert cache.get("k3") == "v3"

    def test_clear(self) -> None:
        cache = _QueryCache()
        cache.put("k1", "v1")
        cache.clear()
        assert cache.get("k1") is None


# ═══════════════════════════════════════════════════════════════════════════════
# generate_with_rag
# ═══════════════════════════════════════════════════════════════════════════════


@pytest.mark.asyncio
async def test_generate_with_rag_basic(rag, mock_llm, mock_knowledge) -> None:
    """Basic non-streaming RAG generation."""
    messages = [ChatMessage(role="user", content="When do backups run?")]

    result = await rag.generate_with_rag(
        messages,
        detected_contexts=["IT"],
        stream=False,
    )

    assert isinstance(result, ChatCompletion)
    assert result.choices[0].message.content  # has content

    # Knowledge search was called
    mock_knowledge.search_knowledge.assert_called_once()

    # LLM was called with enriched messages (system + user)
    call_args = mock_llm.complete.call_args
    enriched_msgs = call_args.args[0]
    assert len(enriched_msgs) == 2  # RAG system + user
    assert enriched_msgs[0].role == "system"
    assert "KNOWLEDGE BASE" in enriched_msgs[0].content


@pytest.mark.asyncio
async def test_generate_with_rag_source_attribution(rag, mock_llm) -> None:
    """Non-streaming response should include source attribution."""
    messages = [ChatMessage(role="user", content="Upload limit?")]

    result = await rag.generate_with_rag(
        messages,
        detected_contexts=["IT"],
        stream=False,
    )

    content = result.choices[0].message.content
    assert "Fonti" in content


@pytest.mark.asyncio
async def test_generate_with_rag_no_sources_when_disabled(
    mock_llm, mock_knowledge, mock_settings
) -> None:
    """show_sources=False suppresses attribution."""
    rag = RAGOrchestrator(
        mock_llm, mock_knowledge, mock_settings, show_sources=False
    )
    messages = [ChatMessage(role="user", content="test")]

    result = await rag.generate_with_rag(messages, ["IT"], stream=False)

    content = result.choices[0].message.content
    assert "Fonti" not in content


@pytest.mark.asyncio
async def test_generate_with_rag_empty_knowledge(rag, mock_llm, mock_knowledge) -> None:
    """When no knowledge is found, pass messages through unchanged."""
    mock_knowledge.search_knowledge.return_value = []
    messages = [ChatMessage(role="user", content="unknown topic")]

    result = await rag.generate_with_rag(messages, ["IT"], stream=False)

    # LLM should get original messages (no system RAG message)
    call_args = mock_llm.complete.call_args
    enriched_msgs = call_args.args[0]
    assert len(enriched_msgs) == 1  # just the user message


@pytest.mark.asyncio
async def test_generate_with_rag_fallback_on_search_error(
    rag, mock_llm, mock_knowledge
) -> None:
    """If knowledge search fails, fall back to plain LLM."""
    mock_knowledge.search_knowledge.side_effect = Exception("vector store down")
    messages = [ChatMessage(role="user", content="test")]

    result = await rag.generate_with_rag(messages, ["IT"], stream=False)

    # Should still get a response (plain LLM without RAG)
    assert isinstance(result, ChatCompletion)
    call_args = mock_llm.complete.call_args
    enriched_msgs = call_args.args[0]
    assert len(enriched_msgs) == 1  # no RAG system message


@pytest.mark.asyncio
async def test_generate_with_rag_caches_queries(rag, mock_llm, mock_knowledge) -> None:
    """Repeated identical queries should use cache."""
    messages = [ChatMessage(role="user", content="same question")]

    await rag.generate_with_rag(messages, ["IT"], stream=False)
    await rag.generate_with_rag(messages, ["IT"], stream=False)

    # Knowledge search should only be called once
    assert mock_knowledge.search_knowledge.call_count == 1


@pytest.mark.asyncio
async def test_generate_with_rag_streaming(rag, mock_llm, mock_knowledge) -> None:
    """Streaming mode should return an async generator."""
    async def fake_stream(*args, **kwargs):
        yield ChatCompletionChunk(
            id="c1", model="m",
            choices=[StreamChoice(delta=DeltaMessage(role="assistant"))],
        )
        yield ChatCompletionChunk(
            id="c1", model="m",
            choices=[StreamChoice(delta=DeltaMessage(content="Hello"))],
        )
        yield ChatCompletionChunk(
            id="c1", model="m",
            choices=[StreamChoice(delta=DeltaMessage(), finish_reason="stop")],
        )

    mock_llm.complete.return_value = fake_stream()

    messages = [ChatMessage(role="user", content="test")]
    gen = await rag.generate_with_rag(messages, ["IT"], stream=True)

    chunks = [chunk async for chunk in gen]

    # Should have original chunks + source attribution chunk
    assert len(chunks) >= 3
    # Last chunk should be source attribution
    last_content = chunks[-1].choices[0].delta.content
    assert "Fonti" in last_content


# ═══════════════════════════════════════════════════════════════════════════════
# build_rag_prompt
# ═══════════════════════════════════════════════════════════════════════════════


@pytest.mark.asyncio
async def test_build_rag_prompt_basic(rag) -> None:
    results = [
        FakeSearchResult(item=FakeKnowledgeItem(content="Fact 1"), score=0.9),
        FakeSearchResult(item=FakeKnowledgeItem(content="Fact 2"), score=0.7),
    ]

    prompt = await rag.build_rag_prompt("query", results)

    assert "KNOWLEDGE BASE" in prompt
    assert "Fact 1" in prompt
    assert "Fact 2" in prompt
    assert "90%" in prompt  # score formatting


@pytest.mark.asyncio
async def test_build_rag_prompt_empty_results(rag) -> None:
    prompt = await rag.build_rag_prompt("query", [])
    assert prompt == ""


@pytest.mark.asyncio
async def test_build_rag_prompt_respects_token_budget(rag) -> None:
    """Very long knowledge should be truncated."""
    long_content = "A" * 10000
    results = [
        FakeSearchResult(item=FakeKnowledgeItem(content=long_content), score=0.9),
    ]

    prompt = await rag.build_rag_prompt("query", results, max_context_tokens=200)

    # Should be way shorter than the full content
    assert len(prompt) < 2000


# ═══════════════════════════════════════════════════════════════════════════════
# extract_and_store_knowledge
# ═══════════════════════════════════════════════════════════════════════════════


@pytest.mark.asyncio
async def test_extract_and_store_basic(rag, mock_llm, mock_knowledge) -> None:
    """Successful extraction stores items."""
    mock_llm.chat.return_value = '[{"content": "New fact", "confidence": 0.9}]'

    conversation = MagicMock()
    conversation.id = "conv-1"
    conversation.messages = []

    items = await rag.extract_and_store_knowledge(
        conversation=conversation,
        assistant_response="The backup runs at 2am.",
        detected_contexts=["IT"],
    )

    assert len(items) == 1
    mock_knowledge.add_knowledge.assert_called_once()
    call_kwargs = mock_knowledge.add_knowledge.call_args.kwargs
    assert call_kwargs["content"] == "New fact"
    assert call_kwargs["contexts"] == ["IT"]
    assert call_kwargs["source_type"] == "conversation_extract"


@pytest.mark.asyncio
async def test_extract_empty_response(rag, mock_llm) -> None:
    """Empty assistant response returns no items."""
    items = await rag.extract_and_store_knowledge(
        conversation=MagicMock(),
        assistant_response="",
        detected_contexts=["IT"],
    )
    assert items == []
    mock_llm.chat.assert_not_called()


@pytest.mark.asyncio
async def test_extract_filters_by_confidence(rag, mock_llm, mock_knowledge) -> None:
    """Low confidence items are filtered out."""
    mock_llm.chat.return_value = (
        '[{"content": "Low conf", "confidence": 0.3}, '
        '{"content": "High conf", "confidence": 0.8}]'
    )

    items = await rag.extract_and_store_knowledge(
        conversation=MagicMock(id="c1", messages=[]),
        assistant_response="Some response",
        detected_contexts=["IT"],
    )

    assert mock_knowledge.add_knowledge.call_count == 1
    assert mock_knowledge.add_knowledge.call_args.kwargs["content"] == "High conf"


@pytest.mark.asyncio
async def test_extract_handles_simple_format(rag, mock_llm, mock_knowledge) -> None:
    """Simple string array format is also supported."""
    mock_llm.chat.return_value = '["Simple fact 1", "Simple fact 2"]'

    items = await rag.extract_and_store_knowledge(
        conversation=MagicMock(id="c1", messages=[]),
        assistant_response="Response with facts",
        detected_contexts=["IT"],
    )

    assert mock_knowledge.add_knowledge.call_count == 2


@pytest.mark.asyncio
async def test_extract_handles_llm_failure(rag, mock_llm) -> None:
    """LLM failure during extraction returns empty."""
    mock_llm.chat.side_effect = Exception("LLM down")

    items = await rag.extract_and_store_knowledge(
        conversation=MagicMock(id="c1"),
        assistant_response="Some response",
        detected_contexts=["IT"],
    )

    assert items == []


@pytest.mark.asyncio
async def test_extract_handles_invalid_json(rag, mock_llm) -> None:
    """Invalid JSON from LLM returns empty."""
    mock_llm.chat.return_value = "Not valid JSON at all"

    items = await rag.extract_and_store_knowledge(
        conversation=MagicMock(id="c1", messages=[]),
        assistant_response="Some response",
        detected_contexts=["IT"],
    )

    assert items == []


# ═══════════════════════════════════════════════════════════════════════════════
# rerank_results
# ═══════════════════════════════════════════════════════════════════════════════


@pytest.mark.asyncio
async def test_rerank_success(rag, mock_llm) -> None:
    """LLM reranking reorders results."""
    results = [
        FakeSearchResult(item=FakeKnowledgeItem(content="A"), score=0.5),
        FakeSearchResult(item=FakeKnowledgeItem(content="B"), score=0.6),
        FakeSearchResult(item=FakeKnowledgeItem(content="C"), score=0.7),
    ]

    mock_llm.chat.return_value = "[2, 0]"  # C first, then A

    reranked = await rag.rerank_results("query", results, top_k=3)

    assert len(reranked) == 2
    assert reranked[0].item.content == "C"
    assert reranked[1].item.content == "A"


@pytest.mark.asyncio
async def test_rerank_fallback_on_failure(rag, mock_llm) -> None:
    """If reranking fails, return original order."""
    results = [
        FakeSearchResult(item=FakeKnowledgeItem(content="A"), score=0.5),
        FakeSearchResult(item=FakeKnowledgeItem(content="B"), score=0.6),
    ]

    mock_llm.chat.side_effect = Exception("LLM down")

    reranked = await rag.rerank_results("query", results, top_k=2)

    assert len(reranked) == 2
    assert reranked[0].item.content == "A"  # original order preserved


@pytest.mark.asyncio
async def test_rerank_single_result(rag) -> None:
    """Single result doesn't need reranking."""
    results = [FakeSearchResult()]
    reranked = await rag.rerank_results("query", results)
    assert len(reranked) == 1


@pytest.mark.asyncio
async def test_rerank_invalid_indices(rag, mock_llm) -> None:
    """Invalid indices are ignored, falls back to original."""
    results = [
        FakeSearchResult(item=FakeKnowledgeItem(content="A")),
        FakeSearchResult(item=FakeKnowledgeItem(content="B")),
    ]

    mock_llm.chat.return_value = "[99, -1]"  # all out of range

    reranked = await rag.rerank_results("query", results)

    # Falls back to original order (empty valid indices → fallback)
    assert len(reranked) == 2


# ═══════════════════════════════════════════════════════════════════════════════
# Internal parsers
# ═══════════════════════════════════════════════════════════════════════════════


class TestParseExtractionResponse:
    def test_structured_format(self) -> None:
        rag = RAGOrchestrator.__new__(RAGOrchestrator)
        result = rag._parse_extraction_response(
            '[{"content": "fact", "confidence": 0.9}]',
            min_confidence=0.5,
        )
        assert len(result) == 1
        assert result[0]["content"] == "fact"

    def test_simple_format(self) -> None:
        rag = RAGOrchestrator.__new__(RAGOrchestrator)
        result = rag._parse_extraction_response(
            '["fact1", "fact2"]',
            min_confidence=0.5,
        )
        assert len(result) == 2

    def test_invalid_json(self) -> None:
        rag = RAGOrchestrator.__new__(RAGOrchestrator)
        result = rag._parse_extraction_response("not json", min_confidence=0.5)
        assert result == []

    def test_json_in_text(self) -> None:
        rag = RAGOrchestrator.__new__(RAGOrchestrator)
        result = rag._parse_extraction_response(
            'Here are the facts: ["fact1"] end.',
            min_confidence=0.5,
        )
        assert len(result) == 1

    def test_filters_low_confidence(self) -> None:
        rag = RAGOrchestrator.__new__(RAGOrchestrator)
        result = rag._parse_extraction_response(
            '[{"content": "low", "confidence": 0.2}, {"content": "high", "confidence": 0.8}]',
            min_confidence=0.5,
        )
        assert len(result) == 1
        assert result[0]["content"] == "high"


class TestParseRerankResponse:
    def test_valid_indices(self) -> None:
        result = RAGOrchestrator._parse_rerank_response("[2, 0, 1]", max_index=3)
        assert result == [2, 0, 1]

    def test_out_of_range(self) -> None:
        result = RAGOrchestrator._parse_rerank_response("[0, 5, 1]", max_index=3)
        assert result == [0, 1]

    def test_duplicates_removed(self) -> None:
        result = RAGOrchestrator._parse_rerank_response("[0, 0, 1]", max_index=3)
        assert result == [0, 1]

    def test_invalid_json(self) -> None:
        result = RAGOrchestrator._parse_rerank_response("invalid", max_index=3)
        assert result == []

    def test_empty_array(self) -> None:
        result = RAGOrchestrator._parse_rerank_response("[]", max_index=3)
        assert result == []


# ═══════════════════════════════════════════════════════════════════════════════
# Cache management
# ═══════════════════════════════════════════════════════════════════════════════


def test_clear_cache(rag) -> None:
    rag._cache.put("k1", "v1")
    rag.clear_cache()
    assert rag._cache.get("k1") is None
