"""Tests for the Ollama LLM provider."""

import json
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

from src.llm.models import (
    ChatCompletion,
    ChatCompletionChunk,
    ChatMessage,
    HealthStatus,
    ModelDetails,
    ModelInfo,
)
from src.llm.ollama import OllamaProvider
from src.shared.exceptions import LLMError


@pytest.fixture
def provider() -> OllamaProvider:
    """Create a provider with test settings."""
    with patch("src.llm.ollama.get_settings") as mock_settings:
        mock_settings.return_value = MagicMock(
            ollama_base_url="http://fake-ollama:11434",
            ollama_model="testmodel:7b",
            ollama_timeout=30,
        )
        return OllamaProvider()


# ═══════════════════════════════════════════════════════════════════════════════
# complete – non-streaming
# ═══════════════════════════════════════════════════════════════════════════════


@pytest.mark.asyncio
async def test_complete_blocking(provider: OllamaProvider) -> None:
    """Non-streaming complete returns a ChatCompletion."""
    response_data = {
        "message": {"role": "assistant", "content": "Hello!"},
        "done": True,
        "prompt_eval_count": 10,
        "eval_count": 5,
    }
    mock_resp = httpx.Response(200, json=response_data, request=httpx.Request("POST", "http://fake"))

    with patch.object(provider, "_retry_request", new_callable=AsyncMock) as mock_req:
        mock_req.return_value = mock_resp
        result = await provider.complete(
            [ChatMessage(role="user", content="Hi")],
            temperature=0.5,
        )

    assert isinstance(result, ChatCompletion)
    assert result.choices[0].message.content == "Hello!"
    assert result.usage.prompt_tokens == 10
    assert result.usage.completion_tokens == 5


@pytest.mark.asyncio
async def test_complete_with_custom_model(provider: OllamaProvider) -> None:
    """Model can be overridden per-request."""
    mock_resp = httpx.Response(
        200,
        json={"message": {"content": "OK"}, "done": True},
        request=httpx.Request("POST", "http://fake"),
    )

    with patch.object(provider, "_retry_request", new_callable=AsyncMock) as mock_req:
        mock_req.return_value = mock_resp
        await provider.complete(
            [ChatMessage(content="test")],
            model="custom:13b",
        )

    payload = mock_req.call_args.kwargs["json"]
    assert payload["model"] == "custom:13b"


# ═══════════════════════════════════════════════════════════════════════════════
# complete – streaming
# ═══════════════════════════════════════════════════════════════════════════════


@pytest.mark.asyncio
async def test_complete_streaming(provider: OllamaProvider) -> None:
    """Streaming complete returns an async generator of ChatCompletionChunk."""
    lines = [
        json.dumps({"message": {"content": "Hel"}, "done": False}),
        json.dumps({"message": {"content": "lo"}, "done": False}),
        json.dumps({"message": {"content": ""}, "done": True}),
    ]

    async def fake_aiter_lines():
        for line in lines:
            yield line

    mock_response = AsyncMock()
    mock_response.raise_for_status = MagicMock()
    mock_response.aiter_lines = fake_aiter_lines

    mock_client = AsyncMock()
    mock_client.stream = MagicMock(return_value=_AsyncContextManager(mock_response))

    with patch.object(provider, "_get_client", return_value=mock_client):
        gen = await provider.complete(
            [ChatMessage(content="Hi")],
            stream=True,
        )
        chunks = [chunk async for chunk in gen]

    # First chunk: role, then 2 content chunks, then stop
    assert len(chunks) >= 3
    assert any(c.choices[0].delta.role == "assistant" for c in chunks)
    content_parts = [c.choices[0].delta.content for c in chunks if c.choices[0].delta.content]
    assert "Hel" in content_parts
    assert "lo" in content_parts


# ═══════════════════════════════════════════════════════════════════════════════
# Legacy backward-compatible methods
# ═══════════════════════════════════════════════════════════════════════════════


@pytest.mark.asyncio
async def test_chat_legacy(provider: OllamaProvider) -> None:
    """Legacy .chat() method returns plain string."""
    mock_resp = httpx.Response(
        200,
        json={"message": {"content": "Legacy!"}, "done": True},
        request=httpx.Request("POST", "http://fake"),
    )

    with patch.object(provider, "_retry_request", new_callable=AsyncMock) as mock_req:
        mock_req.return_value = mock_resp
        result = await provider.chat([{"role": "user", "content": "test"}])

    assert result == "Legacy!"
    assert isinstance(result, str)


@pytest.mark.asyncio
async def test_generate_legacy(provider: OllamaProvider) -> None:
    """Legacy .generate() method returns plain string."""
    mock_resp = httpx.Response(
        200,
        json={"message": {"content": "Generated!"}, "done": True},
        request=httpx.Request("POST", "http://fake"),
    )

    with patch.object(provider, "_retry_request", new_callable=AsyncMock) as mock_req:
        mock_req.return_value = mock_resp
        result = await provider.generate("write a poem")

    assert result == "Generated!"


# ═══════════════════════════════════════════════════════════════════════════════
# embed
# ═══════════════════════════════════════════════════════════════════════════════


@pytest.mark.asyncio
async def test_embed(provider: OllamaProvider) -> None:
    mock_resp = httpx.Response(
        200,
        json={"embeddings": [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]},
        request=httpx.Request("POST", "http://fake"),
    )

    with patch.object(provider, "_retry_request", new_callable=AsyncMock) as mock_req:
        mock_req.return_value = mock_resp
        vectors = await provider.embed(["hello", "world"])

    assert len(vectors) == 2
    assert vectors[0] == [0.1, 0.2, 0.3]


@pytest.mark.asyncio
async def test_embed_fallback(provider: OllamaProvider) -> None:
    """Falls back to single-text /api/embeddings for older Ollama."""
    # First call returns empty embeddings (triggers fallback)
    empty_resp = httpx.Response(
        200,
        json={"embeddings": []},
        request=httpx.Request("POST", "http://fake"),
    )
    single_resp = httpx.Response(
        200,
        json={"embedding": [0.1, 0.2]},
        request=httpx.Request("POST", "http://fake"),
    )

    with patch.object(provider, "_retry_request", new_callable=AsyncMock) as mock_req:
        mock_req.side_effect = [empty_resp, single_resp]
        vectors = await provider.embed(["hello"])

    assert vectors == [[0.1, 0.2]]


# ═══════════════════════════════════════════════════════════════════════════════
# list_models
# ═══════════════════════════════════════════════════════════════════════════════


@pytest.mark.asyncio
async def test_list_models(provider: OllamaProvider) -> None:
    mock_resp = httpx.Response(
        200,
        json={"models": [
            {"name": "llama3:8b"},
            {"name": "phi3:mini"},
        ]},
        request=httpx.Request("GET", "http://fake"),
    )

    with patch.object(provider, "_retry_request", new_callable=AsyncMock) as mock_req:
        mock_req.return_value = mock_resp
        models = await provider.list_models()

    assert len(models) == 2
    assert isinstance(models[0], ModelInfo)
    assert models[0].id == "llama3:8b"
    assert models[0].owned_by == "ollama"


# ═══════════════════════════════════════════════════════════════════════════════
# health_check
# ═══════════════════════════════════════════════════════════════════════════════


@pytest.mark.asyncio
async def test_health_check_healthy(provider: OllamaProvider) -> None:
    mock_client = AsyncMock()
    mock_client.get = AsyncMock(return_value=httpx.Response(200, request=httpx.Request("GET", "http://fake")))
    mock_client.is_closed = False

    with patch.object(provider, "_get_client", return_value=mock_client):
        status = await provider.health_check()

    assert isinstance(status, HealthStatus)
    assert status.healthy is True
    assert status.backend == "ollama"
    assert status.latency_ms >= 0


@pytest.mark.asyncio
async def test_health_check_unhealthy(provider: OllamaProvider) -> None:
    mock_client = AsyncMock()
    mock_client.get = AsyncMock(side_effect=httpx.ConnectError("refused"))
    mock_client.is_closed = False

    with patch.object(provider, "_get_client", return_value=mock_client):
        status = await provider.health_check()

    assert status.healthy is False
    assert "refused" in status.detail


# ═══════════════════════════════════════════════════════════════════════════════
# Ollama-specific methods
# ═══════════════════════════════════════════════════════════════════════════════


@pytest.mark.asyncio
async def test_model_info(provider: OllamaProvider) -> None:
    mock_resp = httpx.Response(
        200,
        json={"details": {
            "family": "llama",
            "parameter_size": "7B",
            "quantization_level": "Q4_0",
            "format": "gguf",
        }},
        request=httpx.Request("POST", "http://fake"),
    )

    with patch.object(provider, "_retry_request", new_callable=AsyncMock) as mock_req:
        mock_req.return_value = mock_resp
        info = await provider.model_info("llama3:7b")

    assert isinstance(info, ModelDetails)
    assert info.family == "llama"
    assert info.parameter_size == "7B"
    assert info.quantization == "Q4_0"


@pytest.mark.asyncio
async def test_generate_with_context(provider: OllamaProvider) -> None:
    mock_resp = httpx.Response(
        200,
        json={"response": "Continuation text"},
        request=httpx.Request("POST", "http://fake"),
    )

    with patch.object(provider, "_retry_request", new_callable=AsyncMock) as mock_req:
        mock_req.return_value = mock_resp
        result = await provider.generate_with_context("continue", context=[1, 2, 3])

    assert result == "Continuation text"
    payload = mock_req.call_args.kwargs["json"]
    assert payload["context"] == [1, 2, 3]


@pytest.mark.asyncio
async def test_delete_model(provider: OllamaProvider) -> None:
    mock_client = AsyncMock()
    mock_client.request = AsyncMock(return_value=httpx.Response(200, request=httpx.Request("DELETE", "http://fake")))
    mock_client.is_closed = False

    with patch.object(provider, "_get_client", return_value=mock_client):
        result = await provider.delete_model("old-model")

    assert result is True


# ═══════════════════════════════════════════════════════════════════════════════
# Error handling
# ═══════════════════════════════════════════════════════════════════════════════


@pytest.mark.asyncio
async def test_complete_raises_llm_error(provider: OllamaProvider) -> None:
    with patch.object(provider, "_retry_request", new_callable=AsyncMock) as mock_req:
        mock_req.side_effect = LLMError("connection refused")
        with pytest.raises(LLMError, match="connection refused"):
            await provider.complete([ChatMessage(content="test")])


# ═══════════════════════════════════════════════════════════════════════════════
# Connection pooling
# ═══════════════════════════════════════════════════════════════════════════════


@pytest.mark.asyncio
async def test_close(provider: OllamaProvider) -> None:
    """Close should be safe even without an open client."""
    await provider.close()
    await provider.close()  # idempotent


# ═══════════════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════════════


class _AsyncContextManager:
    """Fake async context manager for httpx.stream()."""

    def __init__(self, value):
        self._value = value

    async def __aenter__(self):
        return self._value

    async def __aexit__(self, *args):
        pass
