"""Embedding generation using sentence-transformers (runs locally)."""

from functools import lru_cache

import numpy as np

from src.config.settings import get_settings


class EmbeddingProvider:
    """Generates embeddings using a sentence-transformers model."""

    def __init__(self, model_name: str | None = None):
        self._model_name = model_name or get_settings().embedding_model
        self._model = None

    def _load_model(self):
        # Lazy import to avoid loading torch at module import time
        from sentence_transformers import SentenceTransformer

        if self._model is None:
            self._model = SentenceTransformer(self._model_name)
        return self._model

    def embed(self, texts: list[str]) -> list[list[float]]:
        """Generate embeddings for a list of texts."""
        model = self._load_model()
        embeddings = model.encode(texts, normalize_embeddings=True)
        return embeddings.tolist()

    def embed_single(self, text: str) -> list[float]:
        """Generate embedding for a single text."""
        return self.embed([text])[0]

    @property
    def dimension(self) -> int:
        return get_settings().embedding_dimension


@lru_cache
def get_embedding_provider() -> EmbeddingProvider:
    """Return cached embedding provider singleton."""
    return EmbeddingProvider()
