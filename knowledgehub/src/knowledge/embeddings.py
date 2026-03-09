"""
Embedding abstraction supporting local sentence-transformers and Ollama.

Each backend implements the :class:`EmbeddingProvider` ABC.  The factory
:func:`get_embedding_provider` returns the correct implementation based on
the active profile settings.

Features:
  - Full async interface
  - Batch processing for efficiency
  - Retry logic for resilience (Ollama backend)
  - Metrics: elapsed time and dimension logged for every call
"""

from __future__ import annotations

import asyncio
import time
from abc import ABC, abstractmethod
from functools import lru_cache

import httpx

from src.config.logging import get_logger
from src.config.settings import EmbeddingBackend, get_settings
from src.shared.exceptions import KnowledgeHubError

logger = get_logger(__name__)

# Max retries for transient network errors (Ollama backend)
_MAX_RETRIES = 3
_RETRY_BACKOFF = 1.0  # seconds, doubles on each retry


class EmbeddingError(KnowledgeHubError):
    """Raised when embedding generation fails."""


# ═══════════════════════════════════════════════════════════════════════════
# Abstract base
# ═══════════════════════════════════════════════════════════════════════════


class EmbeddingProvider(ABC):
    """Abstract interface for embedding backends."""

    @abstractmethod
    async def embed(self, texts: list[str]) -> list[list[float]]:
        """Generate embeddings for a list of texts (batch)."""

    async def embed_single(self, text: str) -> list[float]:
        """Convenience: embed a single text and return one vector."""
        results = await self.embed([text])
        return results[0]

    @property
    @abstractmethod
    def dimension(self) -> int:
        """Return the embedding dimension for this provider."""


# ═══════════════════════════════════════════════════════════════════════════
# 1. LocalEmbeddings — sentence-transformers (mini PC)
# ═══════════════════════════════════════════════════════════════════════════


class LocalEmbeddings(EmbeddingProvider):
    """Generates embeddings using a local ``sentence-transformers`` model.

    The model is lazily loaded on first use and cached in memory.
    Batch processing is handled by the underlying library.
    """

    def __init__(self, model_name: str | None = None) -> None:
        settings = get_settings()
        self._model_name = model_name or settings.embedding_model
        self._batch_size = settings.embedding_batch_size
        self._dimension_value = settings.embedding_dimension
        self._model = None

    def _load_model(self):
        """Lazily load the model to avoid torch import at module level."""
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(self._model_name)
            actual_dim = self._model.get_sentence_embedding_dimension()
            if actual_dim:
                self._dimension_value = actual_dim
            logger.info(
                "local_embeddings_loaded",
                model=self._model_name,
                dimension=self._dimension_value,
            )
        return self._model

    async def embed(self, texts: list[str]) -> list[list[float]]:
        t0 = time.monotonic()
        loop = asyncio.get_running_loop()

        def _encode():
            model = self._load_model()
            return model.encode(
                texts,
                normalize_embeddings=True,
                batch_size=self._batch_size,
                show_progress_bar=False,
            ).tolist()

        result = await loop.run_in_executor(None, _encode)
        elapsed = time.monotonic() - t0

        logger.info(
            "local_embed",
            texts=len(texts),
            dimension=self._dimension_value,
            elapsed_ms=round(elapsed * 1000, 1),
        )
        return result

    @property
    def dimension(self) -> int:
        return self._dimension_value


# ═══════════════════════════════════════════════════════════════════════════
# 2. OllamaEmbeddings — via Ollama /api/embed endpoint
# ═══════════════════════════════════════════════════════════════════════════


class OllamaEmbeddings(EmbeddingProvider):
    """Generates embeddings using the Ollama ``/api/embed`` endpoint.

    Supports more powerful models like ``nomic-embed-text``.  Includes
    retry logic for transient network errors.
    """

    def __init__(
        self,
        base_url: str | None = None,
        model: str | None = None,
    ) -> None:
        settings = get_settings()
        self._base_url = (base_url or settings.ollama_base_url).rstrip("/")
        self._model = model or settings.ollama_embedding_model
        self._batch_size = settings.embedding_batch_size
        self._dimension_value = settings.embedding_dimension
        self._timeout = settings.ollama_timeout
        self._client = httpx.AsyncClient(
            base_url=self._base_url,
            timeout=httpx.Timeout(self._timeout),
        )
        logger.info(
            "ollama_embeddings_created",
            base_url=self._base_url,
            model=self._model,
        )

    async def _request_with_retry(self, texts: list[str]) -> list[list[float]]:
        """POST to /api/embed with retry logic."""
        last_error: Exception | None = None
        backoff = _RETRY_BACKOFF

        for attempt in range(_MAX_RETRIES):
            try:
                response = await self._client.post(
                    "/api/embed",
                    json={"model": self._model, "input": texts},
                )
                response.raise_for_status()
                data = response.json()

                embeddings = data.get("embeddings", [])
                if not embeddings:
                    raise EmbeddingError(
                        f"Ollama returned empty embeddings for model {self._model}"
                    )

                # Detect dimension from first response
                if embeddings and len(embeddings[0]) != self._dimension_value:
                    self._dimension_value = len(embeddings[0])

                return embeddings

            except (httpx.HTTPStatusError, httpx.ConnectError, httpx.ReadTimeout) as exc:
                last_error = exc
                if attempt < _MAX_RETRIES - 1:
                    logger.warning(
                        "ollama_embed_retry",
                        attempt=attempt + 1,
                        error=str(exc),
                        backoff_s=backoff,
                    )
                    await asyncio.sleep(backoff)
                    backoff *= 2

        raise EmbeddingError(
            f"Ollama embedding failed after {_MAX_RETRIES} attempts: {last_error}"
        )

    async def embed(self, texts: list[str]) -> list[list[float]]:
        t0 = time.monotonic()

        # Process in batches for large inputs
        all_embeddings: list[list[float]] = []
        for i in range(0, len(texts), self._batch_size):
            batch = texts[i : i + self._batch_size]
            batch_embeddings = await self._request_with_retry(batch)
            all_embeddings.extend(batch_embeddings)

        elapsed = time.monotonic() - t0
        logger.info(
            "ollama_embed",
            texts=len(texts),
            dimension=self._dimension_value,
            elapsed_ms=round(elapsed * 1000, 1),
        )
        return all_embeddings

    @property
    def dimension(self) -> int:
        return self._dimension_value


# ═══════════════════════════════════════════════════════════════════════════
# Factory
# ═══════════════════════════════════════════════════════════════════════════


@lru_cache
def get_embedding_provider() -> EmbeddingProvider:
    """Factory: return the correct embedding provider for the active profile."""
    settings = get_settings()
    if settings.embedding_backend == EmbeddingBackend.OLLAMA:
        return OllamaEmbeddings()
    return LocalEmbeddings()
