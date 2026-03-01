"""Knowledge base service – orchestrates documents, embeddings, and vector store."""

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from src.config.logging import get_logger
from src.knowledge.embeddings import get_embedding_provider
from src.knowledge.models import Document
from src.knowledge.vectorstore import SearchResult, get_vector_store
from src.shared.exceptions import NotFoundError
from src.shared.utils import chunk_text

logger = get_logger(__name__)


class KnowledgeService:
    def __init__(self, session: AsyncSession):
        self._session = session
        self._vectorstore = get_vector_store()
        self._embedder = get_embedding_provider()

    async def add_document(
        self,
        title: str,
        content: str,
        metadata: dict | None = None,
    ) -> Document:
        """Ingest a document: store in DB, chunk, embed, and index."""
        metadata = metadata or {}

        doc = Document(title=title, content=content, metadata_json=metadata)
        self._session.add(doc)
        await self._session.flush()

        # Chunk and embed
        from src.config import get_settings

        settings = get_settings()
        chunks = chunk_text(content, chunk_size=settings.chunk_size, overlap=settings.chunk_overlap)
        doc.chunk_count = len(chunks)

        if chunks:
            embeddings = self._embedder.embed(chunks)
            await self._vectorstore.add(
                doc_id=doc.id,
                chunks=chunks,
                embeddings=embeddings,
                metadata={"title": title, **metadata},
            )

        logger.info("document_added", doc_id=doc.id, chunks=len(chunks))
        return doc

    async def get_document(self, document_id: str) -> Document:
        result = await self._session.get(Document, document_id)
        if result is None:
            raise NotFoundError("Document", document_id)
        return result

    async def list_documents(self, skip: int = 0, limit: int = 20) -> tuple[list[Document], int]:
        count_q = select(func.count()).select_from(Document)
        total = (await self._session.execute(count_q)).scalar_one()

        q = select(Document).offset(skip).limit(limit).order_by(Document.created_at.desc())
        result = await self._session.execute(q)
        docs = list(result.scalars().all())
        return docs, total

    async def delete_document(self, document_id: str) -> None:
        doc = await self.get_document(document_id)
        await self._vectorstore.delete(doc.id)
        await self._session.delete(doc)
        logger.info("document_deleted", doc_id=document_id)

    async def search(self, query: str, top_k: int = 5) -> list[SearchResult]:
        """Semantic search: embed query and search the vector store."""
        embedding = self._embedder.embed_single(query)
        results = await self._vectorstore.search(embedding, top_k=top_k)
        return results
