"""
Vector store abstraction supporting ChromaDB (mini) and Qdrant (enterprise).

Each backend implements the :class:`VectorStore` ABC.  The factory
:func:`get_vector_store` returns the correct implementation based on the
active profile settings.
"""

from __future__ import annotations

import asyncio
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

from src.config.logging import get_logger
from src.config.settings import VectorStoreBackend, get_settings
from src.shared.exceptions import VectorStoreError

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class SearchResult:
    """A single vector search result."""

    id: str
    content: str
    score: float
    metadata: dict = field(default_factory=dict)


@dataclass
class DocumentRecord:
    """Stored document representation returned by :meth:`VectorStore.get`."""

    id: str
    content: str
    metadata: dict = field(default_factory=dict)


# ═══════════════════════════════════════════════════════════════════════════
# Abstract base
# ═══════════════════════════════════════════════════════════════════════════


class VectorStore(ABC):
    """Abstract interface for vector store backends."""

    @abstractmethod
    async def add(
        self,
        texts: list[str],
        metadatas: list[dict],
        ids: list[str],
        embeddings: list[list[float]] | None = None,
    ) -> list[str]:
        """Insert texts with metadata and optional pre-computed embeddings.

        Returns the list of stored IDs.
        """

    @abstractmethod
    async def search(
        self,
        query_embedding: list[float],
        n_results: int = 5,
        filter: dict | None = None,
    ) -> list[SearchResult]:
        """Search by embedding vector.  Optional *filter* narrows the scope."""

    @abstractmethod
    async def delete(self, ids: list[str]) -> bool:
        """Delete vectors by their IDs.  Returns ``True`` on success."""

    @abstractmethod
    async def get(self, ids: list[str]) -> list[DocumentRecord]:
        """Retrieve stored documents by ID."""

    @abstractmethod
    async def update(
        self,
        id: str,
        text: str | None = None,
        metadata: dict | None = None,
        embedding: list[float] | None = None,
    ) -> bool:
        """Update a single entry (text, metadata, or both).

        Returns ``True`` on success, ``False`` if ID was not found.
        """


# ═══════════════════════════════════════════════════════════════════════════
# 1. ChromaVectorStore — mini PC profile
# ═══════════════════════════════════════════════════════════════════════════


class ChromaVectorStore(VectorStore):
    """ChromaDB-backed vector store using ``PersistentClient``.

    Designed for local / mini-PC deployments.  Data is persisted in a
    configurable directory.  Collections can be switched per-context via
    :meth:`get_collection`.
    """

    def __init__(
        self,
        persist_dir: str | None = None,
        collection_name: str | None = None,
    ) -> None:
        import chromadb

        settings = get_settings()
        self._persist_dir = persist_dir or settings.chroma_persist_dir
        self._collection_name = collection_name or settings.chroma_collection

        self._client = chromadb.PersistentClient(path=self._persist_dir)
        self._collection = self._client.get_or_create_collection(
            name=self._collection_name,
            metadata={"hnsw:space": "cosine"},
        )
        logger.info(
            "chroma_initialised",
            persist_dir=self._persist_dir,
            collection=self._collection_name,
        )

    # -- helpers -----------------------------------------------------------

    @staticmethod
    async def _run_sync(func, *args, **kwargs):
        """Run a synchronous chromadb call without blocking the event loop."""
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, lambda: func(*args, **kwargs))

    def get_collection(self, name: str):
        """Get or create a named collection (for per-context collections)."""
        return self._client.get_or_create_collection(
            name=name,
            metadata={"hnsw:space": "cosine"},
        )

    # -- VectorStore interface ---------------------------------------------

    async def add(
        self,
        texts: list[str],
        metadatas: list[dict],
        ids: list[str],
        embeddings: list[list[float]] | None = None,
    ) -> list[str]:
        t0 = time.monotonic()
        try:
            kwargs: dict[str, Any] = {
                "ids": ids,
                "documents": texts,
                "metadatas": metadatas,
            }
            if embeddings is not None:
                kwargs["embeddings"] = embeddings

            await self._run_sync(self._collection.add, **kwargs)
            elapsed = time.monotonic() - t0
            logger.info("chroma_add", count=len(ids), elapsed_ms=round(elapsed * 1000, 1))
            return ids
        except Exception as exc:
            raise VectorStoreError(f"ChromaDB add failed: {exc}") from exc

    async def search(
        self,
        query_embedding: list[float],
        n_results: int = 5,
        filter: dict | None = None,
    ) -> list[SearchResult]:
        t0 = time.monotonic()
        try:
            kwargs: dict[str, Any] = {
                "query_embeddings": [query_embedding],
                "n_results": n_results,
                "include": ["documents", "metadatas", "distances"],
            }
            if filter:
                kwargs["where"] = filter

            results = await self._run_sync(self._collection.query, **kwargs)
            elapsed = time.monotonic() - t0

            items: list[SearchResult] = []
            if results["documents"] and results["documents"][0]:
                for doc, meta, dist in zip(
                    results["documents"][0],
                    results["metadatas"][0],
                    results["distances"][0],
                ):
                    items.append(
                        SearchResult(
                            id=meta.get("id", ""),
                            content=doc,
                            score=1.0 - dist,  # cosine distance → similarity
                            metadata=meta,
                        )
                    )

            logger.info(
                "chroma_search",
                results=len(items),
                elapsed_ms=round(elapsed * 1000, 1),
            )
            return items
        except Exception as exc:
            raise VectorStoreError(f"ChromaDB search failed: {exc}") from exc

    async def delete(self, ids: list[str]) -> bool:
        try:
            await self._run_sync(self._collection.delete, ids=ids)
            logger.info("chroma_delete", count=len(ids))
            return True
        except Exception as exc:
            raise VectorStoreError(f"ChromaDB delete failed: {exc}") from exc

    async def get(self, ids: list[str]) -> list[DocumentRecord]:
        try:
            results = await self._run_sync(
                self._collection.get,
                ids=ids,
                include=["documents", "metadatas"],
            )
            records: list[DocumentRecord] = []
            if results["ids"]:
                for rid, doc, meta in zip(
                    results["ids"],
                    results["documents"] or [],
                    results["metadatas"] or [],
                ):
                    records.append(
                        DocumentRecord(id=rid, content=doc or "", metadata=meta or {})
                    )
            return records
        except Exception as exc:
            raise VectorStoreError(f"ChromaDB get failed: {exc}") from exc

    async def update(
        self,
        id: str,
        text: str | None = None,
        metadata: dict | None = None,
        embedding: list[float] | None = None,
    ) -> bool:
        try:
            kwargs: dict[str, Any] = {"ids": [id]}
            if text is not None:
                kwargs["documents"] = [text]
            if metadata is not None:
                kwargs["metadatas"] = [metadata]
            if embedding is not None:
                kwargs["embeddings"] = [embedding]

            await self._run_sync(self._collection.update, **kwargs)
            logger.info("chroma_update", id=id)
            return True
        except Exception as exc:
            raise VectorStoreError(f"ChromaDB update failed: {exc}") from exc


# ═══════════════════════════════════════════════════════════════════════════
# 2. QdrantVectorStore — enterprise profile
# ═══════════════════════════════════════════════════════════════════════════


class QdrantVectorStore(VectorStore):
    """Qdrant-backed vector store using the async ``qdrant-client``.

    Supports multiple collections and advanced payload-based filtering.
    """

    def __init__(
        self,
        host: str | None = None,
        port: int | None = None,
        collection_name: str | None = None,
    ) -> None:
        from qdrant_client import AsyncQdrantClient

        settings = get_settings()
        self._host = host or settings.qdrant_host
        self._port = port or settings.qdrant_port
        self._collection_name = collection_name or settings.qdrant_collection
        self._dimension = settings.embedding_dimension

        self._client = AsyncQdrantClient(host=self._host, port=self._port)
        self._initialised = False

        logger.info(
            "qdrant_client_created",
            host=self._host,
            port=self._port,
            collection=self._collection_name,
        )

    async def _ensure_collection(self) -> None:
        """Create the collection if it doesn't exist yet (idempotent)."""
        if self._initialised:
            return

        from qdrant_client.models import Distance, VectorParams

        collections = await self._client.get_collections()
        names = [c.name for c in collections.collections]

        if self._collection_name not in names:
            await self._client.create_collection(
                collection_name=self._collection_name,
                vectors_config=VectorParams(
                    size=self._dimension,
                    distance=Distance.COSINE,
                ),
            )
            logger.info("qdrant_collection_created", collection=self._collection_name)

        self._initialised = True

    # -- VectorStore interface ---------------------------------------------

    async def add(
        self,
        texts: list[str],
        metadatas: list[dict],
        ids: list[str],
        embeddings: list[list[float]] | None = None,
    ) -> list[str]:
        from qdrant_client.models import PointStruct

        await self._ensure_collection()
        t0 = time.monotonic()

        if embeddings is None:
            raise VectorStoreError("QdrantVectorStore.add requires pre-computed embeddings")

        try:
            points = [
                PointStruct(
                    id=uid,
                    vector=emb,
                    payload={**meta, "content": text},
                )
                for uid, text, meta, emb in zip(ids, texts, metadatas, embeddings)
            ]
            await self._client.upsert(
                collection_name=self._collection_name,
                points=points,
            )
            elapsed = time.monotonic() - t0
            logger.info("qdrant_add", count=len(ids), elapsed_ms=round(elapsed * 1000, 1))
            return ids
        except Exception as exc:
            raise VectorStoreError(f"Qdrant add failed: {exc}") from exc

    async def search(
        self,
        query_embedding: list[float],
        n_results: int = 5,
        filter: dict | None = None,
    ) -> list[SearchResult]:
        await self._ensure_collection()
        t0 = time.monotonic()

        try:
            query_filter = None
            if filter:
                from qdrant_client.models import FieldCondition, Filter, MatchValue

                must_conditions = [
                    FieldCondition(key=k, match=MatchValue(value=v))
                    for k, v in filter.items()
                ]
                query_filter = Filter(must=must_conditions)

            hits = await self._client.search(
                collection_name=self._collection_name,
                query_vector=query_embedding,
                limit=n_results,
                query_filter=query_filter,
            )
            elapsed = time.monotonic() - t0

            items = [
                SearchResult(
                    id=str(hit.id),
                    content=hit.payload.get("content", ""),
                    score=hit.score,
                    metadata={k: v for k, v in hit.payload.items() if k != "content"},
                )
                for hit in hits
            ]

            logger.info(
                "qdrant_search",
                results=len(items),
                elapsed_ms=round(elapsed * 1000, 1),
            )
            return items
        except Exception as exc:
            raise VectorStoreError(f"Qdrant search failed: {exc}") from exc

    async def delete(self, ids: list[str]) -> bool:
        from qdrant_client.models import PointIdsList

        await self._ensure_collection()
        try:
            await self._client.delete(
                collection_name=self._collection_name,
                points_selector=PointIdsList(points=ids),
            )
            logger.info("qdrant_delete", count=len(ids))
            return True
        except Exception as exc:
            raise VectorStoreError(f"Qdrant delete failed: {exc}") from exc

    async def get(self, ids: list[str]) -> list[DocumentRecord]:
        await self._ensure_collection()
        try:
            points = await self._client.retrieve(
                collection_name=self._collection_name,
                ids=ids,
                with_payload=True,
            )
            return [
                DocumentRecord(
                    id=str(p.id),
                    content=p.payload.get("content", ""),
                    metadata={k: v for k, v in p.payload.items() if k != "content"},
                )
                for p in points
            ]
        except Exception as exc:
            raise VectorStoreError(f"Qdrant get failed: {exc}") from exc

    async def update(
        self,
        id: str,
        text: str | None = None,
        metadata: dict | None = None,
        embedding: list[float] | None = None,
    ) -> bool:
        from qdrant_client.models import PointStruct

        await self._ensure_collection()
        try:
            existing = await self._client.retrieve(
                collection_name=self._collection_name,
                ids=[id],
                with_payload=True,
                with_vectors=True,
            )
            if not existing:
                return False

            point = existing[0]
            new_payload = dict(point.payload or {})
            if text is not None:
                new_payload["content"] = text
            if metadata is not None:
                new_payload.update(metadata)

            new_vector = embedding if embedding is not None else point.vector

            await self._client.upsert(
                collection_name=self._collection_name,
                points=[
                    PointStruct(id=id, vector=new_vector, payload=new_payload)
                ],
            )
            logger.info("qdrant_update", id=id)
            return True
        except Exception as exc:
            raise VectorStoreError(f"Qdrant update failed: {exc}") from exc


# ═══════════════════════════════════════════════════════════════════════════
# Factory
# ═══════════════════════════════════════════════════════════════════════════

_vector_store: VectorStore | None = None


def get_vector_store() -> VectorStore:
    """Factory: return the correct vector store for the active profile.

    The instance is cached as a module-level singleton so that the same
    client / connection pool is reused across requests.
    """
    global _vector_store
    if _vector_store is not None:
        return _vector_store

    settings = get_settings()
    if settings.vectorstore_backend == VectorStoreBackend.QDRANT:
        _vector_store = QdrantVectorStore()
    else:
        _vector_store = ChromaVectorStore()

    return _vector_store


def reset_vector_store() -> None:
    """Reset the cached singleton (useful in tests)."""
    global _vector_store
    _vector_store = None
