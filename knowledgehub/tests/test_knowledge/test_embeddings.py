"""Tests for the Embedding abstraction.

Tests the EmbeddingProvider ABC interface and the OllamaEmbeddings class
with a mocked HTTP backend.  LocalEmbeddings is tested only for structure
(not the actual model, since sentence-transformers is heavy).
"""

import pytest
import httpx

from src.knowledge.embeddings import (
    EmbeddingError,
    EmbeddingProvider,
    LocalEmbeddings,
    OllamaEmbeddings,
)


# ── ABC ────────────────────────────────────────────────────────────────────


def test_embedding_provider_is_abstract():
    """Cannot instantiate EmbeddingProvider directly."""
    with pytest.raises(TypeError):
        EmbeddingProvider()


# ── LocalEmbeddings structure ──────────────────────────────────────────────


def test_local_embeddings_init():
    """LocalEmbeddings can be created without loading the model."""
    provider = LocalEmbeddings(model_name="all-MiniLM-L6-v2")
    assert provider.dimension > 0
    assert provider._model is None  # lazy-loaded


# ── OllamaEmbeddings ──────────────────────────────────────────────────────


class FakeTransport(httpx.AsyncBaseTransport):
    """Mock transport that returns fake embeddings."""

    def __init__(self, dim: int = 768, fail_times: int = 0):
        self._dim = dim
        self._fail_count = 0
        self._fail_times = fail_times

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        import json

        if self._fail_count < self._fail_times:
            self._fail_count += 1
            return httpx.Response(500, text="Server Error")

        body = json.loads(request.content)
        texts = body.get("input", [])
        embeddings = [[0.1] * self._dim for _ in texts]
        return httpx.Response(200, json={"embeddings": embeddings})


@pytest.fixture
def ollama_provider():
    """Create an OllamaEmbeddings with a mocked HTTP transport."""
    provider = OllamaEmbeddings(base_url="http://fake-ollama:11434", model="test-embed")
    # Replace the httpx client with one using our fake transport
    provider._client = httpx.AsyncClient(
        transport=FakeTransport(dim=768),
        base_url="http://fake-ollama:11434",
    )
    return provider


@pytest.fixture
def ollama_provider_retry():
    """Provider that fails once then succeeds."""
    provider = OllamaEmbeddings(base_url="http://fake-ollama:11434", model="test-embed")
    provider._client = httpx.AsyncClient(
        transport=FakeTransport(dim=768, fail_times=1),
        base_url="http://fake-ollama:11434",
    )
    return provider


@pytest.fixture
def ollama_provider_always_fail():
    """Provider that always fails (exceeds retries)."""
    provider = OllamaEmbeddings(base_url="http://fake-ollama:11434", model="test-embed")
    provider._client = httpx.AsyncClient(
        transport=FakeTransport(dim=768, fail_times=10),
        base_url="http://fake-ollama:11434",
    )
    return provider


@pytest.mark.asyncio
async def test_ollama_embed_single(ollama_provider):
    result = await ollama_provider.embed_single("hello world")
    assert len(result) == 768
    assert all(isinstance(v, float) for v in result)


@pytest.mark.asyncio
async def test_ollama_embed_batch(ollama_provider):
    texts = ["text one", "text two", "text three"]
    results = await ollama_provider.embed(texts)
    assert len(results) == 3
    assert all(len(v) == 768 for v in results)


@pytest.mark.asyncio
async def test_ollama_dimension_detected(ollama_provider):
    await ollama_provider.embed(["test"])
    assert ollama_provider.dimension == 768


@pytest.mark.asyncio
async def test_ollama_retry_succeeds(ollama_provider_retry):
    """First request fails (500), second should succeed."""
    result = await ollama_provider_retry.embed(["retry test"])
    assert len(result) == 1
    assert len(result[0]) == 768


@pytest.mark.asyncio
async def test_ollama_all_retries_fail(ollama_provider_always_fail):
    """All retries exhausted should raise EmbeddingError."""
    with pytest.raises(EmbeddingError, match="failed after"):
        await ollama_provider_always_fail.embed(["fail"])


@pytest.mark.asyncio
async def test_ollama_batch_processing(ollama_provider):
    """Large input is split into batches transparently."""
    ollama_provider._batch_size = 2
    texts = ["a", "b", "c", "d", "e"]
    results = await ollama_provider.embed(texts)
    assert len(results) == 5
