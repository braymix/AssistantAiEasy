"""Vector store abstraction supporting ChromaDB and Qdrant."""

from abc import ABC, abstractmethod
from dataclasses import dataclass

from src.config.settings import VectorStoreBackend, get_settings


@dataclass
class SearchResult:
    """A single vector search result."""

    id: str
    content: str
    title: str
    score: float
    metadata: dict


class VectorStore(ABC):
    """Abstract interface for vector store backends."""

    @abstractmethod
    async def add(self, doc_id: str, chunks: list[str], embeddings: list[list[float]], metadata: dict) -> None:
        """Add document chunks with their embeddings."""

    @abstractmethod
    async def search(self, embedding: list[float], top_k: int = 5) -> list[SearchResult]:
        """Search for similar documents by embedding vector."""

    @abstractmethod
    async def delete(self, doc_id: str) -> None:
        """Delete all chunks for a document."""


class ChromaVectorStore(VectorStore):
    """ChromaDB-backed vector store (mini profile)."""

    def __init__(self):
        import chromadb

        settings = get_settings()
        self._client = chromadb.HttpClient(
            host=settings.chroma_host,
            port=settings.chroma_port,
        )
        self._collection = self._client.get_or_create_collection(
            name=settings.chroma_collection,
            metadata={"hnsw:space": "cosine"},
        )

    async def add(self, doc_id: str, chunks: list[str], embeddings: list[list[float]], metadata: dict) -> None:
        ids = [f"{doc_id}_{i}" for i in range(len(chunks))]
        metadatas = [{**metadata, "doc_id": doc_id, "chunk_index": i} for i in range(len(chunks))]
        self._collection.add(
            ids=ids,
            documents=chunks,
            embeddings=embeddings,
            metadatas=metadatas,
        )

    async def search(self, embedding: list[float], top_k: int = 5) -> list[SearchResult]:
        results = self._collection.query(
            query_embeddings=[embedding],
            n_results=top_k,
            include=["documents", "metadatas", "distances"],
        )
        items: list[SearchResult] = []
        if results["documents"] and results["documents"][0]:
            for doc, meta, dist in zip(
                results["documents"][0],
                results["metadatas"][0],
                results["distances"][0],
            ):
                items.append(
                    SearchResult(
                        id=meta.get("doc_id", ""),
                        content=doc,
                        title=meta.get("title", ""),
                        score=1.0 - dist,  # Convert distance to similarity
                        metadata=meta,
                    )
                )
        return items

    async def delete(self, doc_id: str) -> None:
        self._collection.delete(where={"doc_id": doc_id})


class QdrantVectorStore(VectorStore):
    """Qdrant-backed vector store (full profile)."""

    def __init__(self):
        from qdrant_client import QdrantClient
        from qdrant_client.models import Distance, VectorParams

        settings = get_settings()
        self._client = QdrantClient(
            host=settings.qdrant_host,
            port=settings.qdrant_port,
        )
        self._collection_name = settings.qdrant_collection

        # Ensure collection exists
        collections = [c.name for c in self._client.get_collections().collections]
        if self._collection_name not in collections:
            self._client.create_collection(
                collection_name=self._collection_name,
                vectors_config=VectorParams(
                    size=settings.embedding_dimension,
                    distance=Distance.COSINE,
                ),
            )

    async def add(self, doc_id: str, chunks: list[str], embeddings: list[list[float]], metadata: dict) -> None:
        from qdrant_client.models import PointStruct

        points = [
            PointStruct(
                id=f"{doc_id}_{i}",
                vector=emb,
                payload={**metadata, "doc_id": doc_id, "chunk_index": i, "content": chunk},
            )
            for i, (chunk, emb) in enumerate(zip(chunks, embeddings))
        ]
        self._client.upsert(collection_name=self._collection_name, points=points)

    async def search(self, embedding: list[float], top_k: int = 5) -> list[SearchResult]:
        results = self._client.search(
            collection_name=self._collection_name,
            query_vector=embedding,
            limit=top_k,
        )
        return [
            SearchResult(
                id=hit.payload.get("doc_id", ""),
                content=hit.payload.get("content", ""),
                title=hit.payload.get("title", ""),
                score=hit.score,
                metadata=hit.payload,
            )
            for hit in results
        ]

    async def delete(self, doc_id: str) -> None:
        from qdrant_client.models import Filter, FieldCondition, MatchValue

        self._client.delete(
            collection_name=self._collection_name,
            points_selector=Filter(
                must=[FieldCondition(key="doc_id", match=MatchValue(value=doc_id))]
            ),
        )


def get_vector_store() -> VectorStore:
    """Factory: return the correct vector store for the active profile."""
    settings = get_settings()
    if settings.vectorstore_backend == VectorStoreBackend.QDRANT:
        return QdrantVectorStore()
    return ChromaVectorStore()
