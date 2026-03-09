"""Tests for the vLLM provider."""

import json
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

from src.llm.models import (
    ChatCompletion,
    ChatCompletionChunk,
    ChatMessage,
    HealthStatus,
    ModelInfo,
)
from src.llm.vllm import VLLMProvider
from src.shared.exceptions import LLMError


@pytest.fixture
def provider() -> VLLMProvider:
    with patch("src.llm.vllm.get_settings") as mock_settings:
        mock_settings.return_value = MagicMock(
            vllm_base_url="http://fake-vllm:8000",
            vllm_model="test-model",
            vllm_max_tokens=2048,
        )
        return VLLMProvider()


# ═══════════════════════════════════════════════════════════════════════════════
# complete – non-streaming
# ═══════════════════════════════════════════════════════════════════════════════


@pytest.mark.asyncio
async def test_complete_blocking(provider: VLLMProvider) -> None:
    response_data = {
        "id": "chatcmpl-test",
        "choices": [{"index": 0, "message": {"content": "Hello!"}, "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
    }
    mock_resp = httpx.Response(200, json=response_data, request=httpx.Request("POST", "http://fake"))

    with patch.object(provider, "_retry_request", new_callable=AsyncMock) as mock_req:
        mock_req.return_value = mock_resp
        result = await provider.complete(
            [ChatMessage(role="user", content="Hi")],
            temperature=0.5,
            max_tokens=100,
        )

    assert isinstance(result, ChatCompletion)
    assert result.choices[0].message.content == "Hello!"
    assert result.usage.prompt_tokens == 10
    assert result.usage.total_tokens == 15


@pytest.mark.asyncio
async def test_complete_with_lora_adapter(provider: VLLMProvider) -> None:
    """LoRA adapter can be selected via model parameter."""
    mock_resp = httpx.Response(
        200,
        json={"choices": [{"message": {"content": "OK"}}], "usage": {}},
        request=httpx.Request("POST", "http://fake"),
    )

    with patch.object(provider, "_retry_request", new_callable=AsyncMock) as mock_req:
        mock_req.return_value = mock_resp
        await provider.complete(
            [ChatMessage(content="test")],
            model="my-lora-adapter",
        )

    payload = mock_req.call_args.kwargs["json"]
    assert payload["model"] == "my-lora-adapter"


@pytest.mark.asyncio
async def test_complete_forwards_optional_params(provider: VLLMProvider) -> None:
    """Extra kwargs like top_p, seed are forwarded."""
    mock_resp = httpx.Response(
        200,
        json={"choices": [{"message": {"content": "OK"}}], "usage": {}},
        request=httpx.Request("POST", "http://fake"),
    )

    with patch.object(provider, "_retry_request", new_callable=AsyncMock) as mock_req:
        mock_req.return_value = mock_resp
        await provider.complete(
            [ChatMessage(content="test")],
            top_p=0.9,
            seed=42,
        )

    payload = mock_req.call_args.kwargs["json"]
    assert payload["top_p"] == 0.9
    assert payload["seed"] == 42


# ═══════════════════════════════════════════════════════════════════════════════
# complete – streaming
# ═══════════════════════════════════════════════════════════════════════════════


@pytest.mark.asyncio
async def test_complete_streaming(provider: VLLMProvider) -> None:
    stop_data = {"choices": [{"delta": {}, "finish_reason": "stop"}]}
    lines = [
        f'data: {json.dumps({"choices": [{"delta": {"role": "assistant"}}]})}',
        f'data: {json.dumps({"choices": [{"delta": {"content": "Hel"}}]})}',
        f'data: {json.dumps({"choices": [{"delta": {"content": "lo"}}]})}',
        f"data: {json.dumps(stop_data)}",
        "data: [DONE]",
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

    assert len(chunks) >= 3
    contents = [c.choices[0].delta.content for c in chunks if c.choices[0].delta.content]
    assert "Hel" in contents
    assert "lo" in contents


# ═══════════════════════════════════════════════════════════════════════════════
# Legacy methods
# ═══════════════════════════════════════════════════════════════════════════════


@pytest.mark.asyncio
async def test_chat_legacy(provider: VLLMProvider) -> None:
    mock_resp = httpx.Response(
        200,
        json={"choices": [{"message": {"content": "Legacy!"}}], "usage": {}},
        request=httpx.Request("POST", "http://fake"),
    )

    with patch.object(provider, "_retry_request", new_callable=AsyncMock) as mock_req:
        mock_req.return_value = mock_resp
        result = await provider.chat([{"role": "user", "content": "test"}])

    assert result == "Legacy!"
    assert isinstance(result, str)


# ═══════════════════════════════════════════════════════════════════════════════
# embed
# ═══════════════════════════════════════════════════════════════════════════════


@pytest.mark.asyncio
async def test_embed(provider: VLLMProvider) -> None:
    mock_resp = httpx.Response(
        200,
        json={"data": [
            {"index": 0, "embedding": [0.1, 0.2]},
            {"index": 1, "embedding": [0.3, 0.4]},
        ]},
        request=httpx.Request("POST", "http://fake"),
    )

    with patch.object(provider, "_retry_request", new_callable=AsyncMock) as mock_req:
        mock_req.return_value = mock_resp
        vectors = await provider.embed(["hello", "world"])

    assert len(vectors) == 2
    assert vectors[0] == [0.1, 0.2]


@pytest.mark.asyncio
async def test_embed_sorted_by_index(provider: VLLMProvider) -> None:
    """Embeddings are sorted by index even if returned out of order."""
    mock_resp = httpx.Response(
        200,
        json={"data": [
            {"index": 1, "embedding": [0.3, 0.4]},
            {"index": 0, "embedding": [0.1, 0.2]},
        ]},
        request=httpx.Request("POST", "http://fake"),
    )

    with patch.object(provider, "_retry_request", new_callable=AsyncMock) as mock_req:
        mock_req.return_value = mock_resp
        vectors = await provider.embed(["a", "b"])

    assert vectors[0] == [0.1, 0.2]
    assert vectors[1] == [0.3, 0.4]


@pytest.mark.asyncio
async def test_embed_batch(provider: VLLMProvider) -> None:
    """embed_batch splits large lists into chunks."""
    call_count = 0

    async def mock_embed(texts, *, model=None):
        nonlocal call_count
        call_count += 1
        return [[0.1] * 3 for _ in texts]

    with patch.object(provider, "embed", side_effect=mock_embed):
        result = await provider.embed_batch(["t"] * 150, batch_size=64)

    assert len(result) == 150
    assert call_count == 3  # 64 + 64 + 22


# ═══════════════════════════════════════════════════════════════════════════════
# list_models
# ═══════════════════════════════════════════════════════════════════════════════


@pytest.mark.asyncio
async def test_list_models(provider: VLLMProvider) -> None:
    mock_resp = httpx.Response(
        200,
        json={"data": [
            {"id": "base-model", "owned_by": "vllm"},
            {"id": "lora-adapter-1", "owned_by": "vllm"},
        ]},
        request=httpx.Request("GET", "http://fake"),
    )

    with patch.object(provider, "_retry_request", new_callable=AsyncMock) as mock_req:
        mock_req.return_value = mock_resp
        models = await provider.list_models()

    assert len(models) == 2
    assert isinstance(models[0], ModelInfo)
    assert models[1].id == "lora-adapter-1"


# ═══════════════════════════════════════════════════════════════════════════════
# health_check
# ═══════════════════════════════════════════════════════════════════════════════


@pytest.mark.asyncio
async def test_health_check_healthy(provider: VLLMProvider) -> None:
    mock_client = AsyncMock()
    mock_client.get = AsyncMock(return_value=httpx.Response(200, request=httpx.Request("GET", "http://fake")))
    mock_client.is_closed = False

    with patch.object(provider, "_get_client", return_value=mock_client):
        status = await provider.health_check()

    assert isinstance(status, HealthStatus)
    assert status.healthy is True
    assert status.backend == "vllm"


@pytest.mark.asyncio
async def test_health_check_unhealthy(provider: VLLMProvider) -> None:
    mock_client = AsyncMock()
    mock_client.get = AsyncMock(side_effect=httpx.ConnectError("refused"))
    mock_client.is_closed = False

    with patch.object(provider, "_get_client", return_value=mock_client):
        status = await provider.health_check()

    assert status.healthy is False
    assert "refused" in status.detail


# ═══════════════════════════════════════════════════════════════════════════════
# Error handling
# ═══════════════════════════════════════════════════════════════════════════════


@pytest.mark.asyncio
async def test_complete_raises_llm_error(provider: VLLMProvider) -> None:
    with patch.object(provider, "_retry_request", new_callable=AsyncMock) as mock_req:
        mock_req.side_effect = LLMError("server down")
        with pytest.raises(LLMError, match="server down"):
            await provider.complete([ChatMessage(content="test")])


@pytest.mark.asyncio
async def test_close(provider: VLLMProvider) -> None:
    await provider.close()
    await provider.close()  # idempotent


# ═══════════════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════════════


class _AsyncContextManager:
    def __init__(self, value):
        self._value = value

    async def __aenter__(self):
        return self._value

    async def __aexit__(self, *args):
        pass
