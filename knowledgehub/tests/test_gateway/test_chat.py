"""Tests for chat completions endpoint and schemas.

Covers:
  - Schema validation (ChatCompletionRequest, Response, Chunks)
  - POST /v1/chat/completions (non-streaming + streaming)
  - RAG context injection
  - Multi-context detection
  - Error handling
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from httpx import ASGITransport, AsyncClient
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from src.gateway.schemas.chat import (
    ChatCompletionChunk,
    ChatCompletionRequest,
    ChatCompletionResponse,
    ChatMessage,
    Choice,
    ChoiceMessage,
    DeltaMessage,
    StreamChoice,
)


# ═══════════════════════════════════════════════════════════════════════════════
# Schema tests
# ═══════════════════════════════════════════════════════════════════════════════


def test_chat_completion_request_defaults():
    req = ChatCompletionRequest(
        messages=[ChatMessage(role="user", content="Hello")]
    )
    assert req.model == ""
    assert req.temperature == 0.7
    assert req.stream is False
    assert len(req.messages) == 1
    assert req.messages[0].role == "user"


def test_chat_completion_request_full():
    req = ChatCompletionRequest(
        model="llama3.2:3b",
        messages=[
            ChatMessage(role="system", content="You are helpful"),
            ChatMessage(role="user", content="What is Python?"),
        ],
        temperature=0.5,
        max_tokens=100,
        stream=True,
        user="test-user",
    )
    assert req.model == "llama3.2:3b"
    assert req.stream is True
    assert req.max_tokens == 100
    assert len(req.messages) == 2


def test_chat_completion_response_structure():
    resp = ChatCompletionResponse(
        model="test-model",
        choices=[Choice(message=ChoiceMessage(content="Hello!"))],
    )
    assert resp.object == "chat.completion"
    assert resp.id.startswith("chatcmpl-")
    assert resp.choices[0].message.role == "assistant"
    assert resp.choices[0].message.content == "Hello!"
    assert resp.choices[0].finish_reason == "stop"


def test_streaming_chunk_structure():
    chunk = ChatCompletionChunk(
        id="chatcmpl-test",
        model="test-model",
        choices=[StreamChoice(delta=DeltaMessage(content="Hi"))],
    )
    assert chunk.object == "chat.completion.chunk"
    assert chunk.choices[0].delta.content == "Hi"
    assert chunk.choices[0].finish_reason is None


def test_streaming_stop_chunk():
    chunk = ChatCompletionChunk(
        id="chatcmpl-test",
        model="test-model",
        choices=[StreamChoice(delta=DeltaMessage(), finish_reason="stop")],
    )
    assert chunk.choices[0].finish_reason == "stop"
    assert chunk.choices[0].delta.content is None


# ═══════════════════════════════════════════════════════════════════════════════
# Endpoint tests
# ═══════════════════════════════════════════════════════════════════════════════


@pytest.mark.asyncio
async def test_list_models(client: AsyncClient):
    response = await client.get("/v1/models")
    assert response.status_code == 200
    data = response.json()
    assert data["object"] == "list"
    assert len(data["data"]) >= 1
    assert "id" in data["data"][0]
    assert data["data"][0]["object"] == "model"


# ── Helper to create a test client with overridden DB + LLM ───────────────

async def _make_test_client(db_engine, llm_mock, detect_patch=None):
    """Create an HTTPX AsyncClient with overridden dependencies."""
    factory = async_sessionmaker(bind=db_engine, class_=AsyncSession, expire_on_commit=False)

    async def _override_session():
        async with factory() as session:
            yield session

    from src.gateway.main import app
    from src.shared.database import get_db_session

    app.dependency_overrides[get_db_session] = _override_session
    return app


@pytest.mark.asyncio
async def test_chat_completion_basic(db_engine, mock_llm_provider):
    """Non-streaming chat completion returns a valid OpenAI-compatible response."""
    app = await _make_test_client(db_engine, mock_llm_provider)

    with patch("src.gateway.routes.chat.get_llm_provider", return_value=mock_llm_provider):
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as ac:
            resp = await ac.post("/v1/chat/completions", json={
                "messages": [{"role": "user", "content": "Hello"}],
                "model": "test-model",
            })

    app.dependency_overrides.clear()

    assert resp.status_code == 200
    data = resp.json()
    assert data["object"] == "chat.completion"
    assert len(data["choices"]) == 1
    assert data["choices"][0]["message"]["role"] == "assistant"
    assert len(data["choices"][0]["message"]["content"]) > 0


@pytest.mark.asyncio
async def test_chat_completion_streaming(db_engine, mock_llm_provider):
    """Streaming chat completion returns SSE chunks ending with [DONE]."""
    app = await _make_test_client(db_engine, mock_llm_provider)

    with patch("src.gateway.routes.chat.get_llm_provider", return_value=mock_llm_provider):
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as ac:
            resp = await ac.post("/v1/chat/completions", json={
                "messages": [{"role": "user", "content": "Hello"}],
                "stream": True,
            })

    app.dependency_overrides.clear()

    assert resp.status_code == 200
    body = resp.text
    assert "data: " in body
    assert "[DONE]" in body


@pytest.mark.asyncio
async def test_chat_with_rag_context(db_engine, mock_llm_provider):
    """When detection triggers, RAG context should be injected into LLM call."""
    app = await _make_test_client(db_engine, mock_llm_provider)

    enriched = [
        {"role": "system", "content": "KNOWLEDGE: Database uses PostgreSQL"},
        {"role": "user", "content": "How do I query the database?"},
    ]

    with (
        patch("src.gateway.routes.chat.get_llm_provider", return_value=mock_llm_provider),
        patch(
            "src.gateway.routes.chat._detect_and_enrich",
            return_value=(enriched, ["database"]),
        ),
    ):
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as ac:
            resp = await ac.post("/v1/chat/completions", json={
                "messages": [{"role": "user", "content": "How do I query the database?"}],
            })

    app.dependency_overrides.clear()

    assert resp.status_code == 200
    mock_llm_provider.chat.assert_called_once()
    call_args = mock_llm_provider.chat.call_args
    assert any("KNOWLEDGE" in str(m) for m in call_args[0][0])


@pytest.mark.asyncio
async def test_chat_no_rules_match(db_engine, mock_llm_provider):
    """When no rules match, messages should pass through unchanged."""
    app = await _make_test_client(db_engine, mock_llm_provider)

    original_msgs = [{"role": "user", "content": "What is the weather today?"}]

    with (
        patch("src.gateway.routes.chat.get_llm_provider", return_value=mock_llm_provider),
        patch(
            "src.gateway.routes.chat._detect_and_enrich",
            return_value=(original_msgs, []),
        ),
    ):
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as ac:
            resp = await ac.post("/v1/chat/completions", json={
                "messages": [{"role": "user", "content": "What is the weather today?"}],
            })

    app.dependency_overrides.clear()

    assert resp.status_code == 200
    mock_llm_provider.chat.assert_called_once()


@pytest.mark.asyncio
async def test_chat_multiple_contexts(db_engine, mock_llm_provider):
    """Multiple contexts can be detected simultaneously."""
    app = await _make_test_client(db_engine, mock_llm_provider)

    enriched = [
        {"role": "system", "content": "KNOWLEDGE: DB + errors context"},
        {"role": "user", "content": "Database query gave ERR-42"},
    ]

    with (
        patch("src.gateway.routes.chat.get_llm_provider", return_value=mock_llm_provider),
        patch(
            "src.gateway.routes.chat._detect_and_enrich",
            return_value=(enriched, ["database", "errors"]),
        ),
    ):
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as ac:
            resp = await ac.post("/v1/chat/completions", json={
                "messages": [{"role": "user", "content": "Database query gave ERR-42"}],
            })

    app.dependency_overrides.clear()

    assert resp.status_code == 200


@pytest.mark.asyncio
async def test_chat_error_handling(db_engine):
    """LLM errors should be caught and returned gracefully."""
    from src.shared.exceptions import LLMError

    app = await _make_test_client(db_engine, None)

    error_llm = AsyncMock()
    error_llm.chat = AsyncMock(side_effect=LLMError("Connection refused"))

    with (
        patch("src.gateway.routes.chat.get_llm_provider", return_value=error_llm),
        patch(
            "src.gateway.routes.chat._detect_and_enrich",
            return_value=([{"role": "user", "content": "Hello"}], []),
        ),
    ):
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as ac:
            resp = await ac.post("/v1/chat/completions", json={
                "messages": [{"role": "user", "content": "Hello"}],
            })

    app.dependency_overrides.clear()

    assert resp.status_code == 200
    data = resp.json()
    assert "[Error:" in data["choices"][0]["message"]["content"]
