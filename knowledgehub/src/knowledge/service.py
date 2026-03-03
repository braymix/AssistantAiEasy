"""
Knowledge Service – full orchestration of the knowledge base.

Coordinates the vector store, embeddings, LLM, and database to provide:
  1. add_knowledge          – embed + store + persist a knowledge item
  2. search_knowledge       – semantic search with context filtering & scoring
  3. extract_knowledge_from_conversation – LLM-driven extraction from chat
  4. build_rag_context      – assemble RAG context for the chat proxy
  5. verify_knowledge       – admin approve/reject workflow
  6. bulk_import            – batch ingest with chunking
  7. add_document / get_document / list_documents / delete_document – doc CRUD
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from fastapi import Depends
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from src.config.logging import get_logger
from src.config.settings import get_settings
from src.knowledge.embeddings import EmbeddingProvider, get_embedding_provider
from src.knowledge.vectorstore import SearchResult, VectorStore, get_vector_store
from src.shared.database import get_db_session
from src.shared.exceptions import NotFoundError
from src.shared.models import (
    ContentType,
    Conversation,
    Document,
    KnowledgeItem,
    Message,
    MessageRole,
)
from src.shared.utils import chunk_text, content_hash

if TYPE_CHECKING:
    from src.llm.base import LLMProvider

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class KnowledgeSearchResult:
    """Enriched search result with DB-level metadata."""

    item: KnowledgeItem
    score: float
    highlights: list[str] = field(default_factory=list)


@dataclass
class ImportResult:
    """Summary of a bulk-import operation."""

    total: int
    imported: int
    errors: list[str] = field(default_factory=list)


@dataclass
class BulkDocument:
    """Input document for :meth:`KnowledgeService.bulk_import`."""

    content: str
    metadata: dict = field(default_factory=dict)
    source: str = "import"


# ---------------------------------------------------------------------------
# LLM extraction prompt
# ---------------------------------------------------------------------------

_EXTRACT_SYSTEM_PROMPT = """\
You are a knowledge extraction assistant. Analyse the conversation below and \
extract distinct, self-contained facts, procedures, or decisions that are worth \
saving in a knowledge base.

Rules:
- Each extracted item MUST be a complete, stand-alone statement.
- Omit greetings, chitchat, and filler.
- Return ONLY a JSON array of strings, e.g. ["fact 1", "fact 2"].
- If nothing is worth extracting, return an empty array: []
"""

_EXTRACT_USER_TEMPLATE = """\
Conversation:
{conversation}

Extract knowledge items as a JSON array of strings:"""


# ═══════════════════════════════════════════════════════════════════════════
# Service
# ═══════════════════════════════════════════════════════════════════════════


class KnowledgeService:
    """Orchestrates documents, embeddings, vector store, and LLM extraction."""

    def __init__(
        self,
        session: AsyncSession,
        vectorstore: VectorStore | None = None,
        embedder: EmbeddingProvider | None = None,
    ) -> None:
        self._session = session
        self._vectorstore = vectorstore or get_vector_store()
        self._embedder = embedder or get_embedding_provider()

    # ------------------------------------------------------------------
    # 1. add_knowledge
    # ------------------------------------------------------------------

    async def add_knowledge(
        self,
        content: str,
        contexts: list[str],
        source_type: str = "manual",
        source_message_id: str | None = None,
        metadata: dict | None = None,
    ) -> KnowledgeItem:
        """Create a knowledge item: embed, store in vector store, persist in DB.

        Performs content-hash deduplication – if an item with the same hash
        already exists, returns the existing one instead.
        """
        metadata = metadata or {}

        # --- deduplication by content hash --------------------------------
        c_hash = content_hash(content)
        existing = await self._session.execute(
            select(KnowledgeItem).where(
                KnowledgeItem.content == content
            ).limit(1)
        )
        duplicate = existing.scalar_one_or_none()
        if duplicate is not None:
            logger.info("knowledge_deduplicated", existing_id=duplicate.id)
            return duplicate

        # --- resolve content type -----------------------------------------
        try:
            ct = ContentType(source_type)
        except ValueError:
            ct = ContentType.MANUAL

        # --- embed --------------------------------------------------------
        embeddings = await self._embedder.embed([content])
        embedding_vector = embeddings[0]
        vec_id = str(uuid.uuid4())

        await self._vectorstore.add(
            texts=[content],
            metadatas=[{
                "contexts": contexts,
                "source_type": source_type,
                "content_hash": c_hash,
                **metadata,
            }],
            ids=[vec_id],
            embeddings=[embedding_vector],
        )

        # --- persist in DB ------------------------------------------------
        item = KnowledgeItem(
            content=content,
            content_type=ct,
            contexts=contexts,
            embedding_id=vec_id,
            source_message_id=source_message_id,
            verified=False,
            created_by=metadata.get("created_by", "system"),
        )
        self._session.add(item)
        await self._session.flush()

        logger.info(
            "knowledge_added",
            item_id=item.id,
            contexts=contexts,
            source_type=source_type,
        )
        return item

    # ------------------------------------------------------------------
    # 2. search_knowledge
    # ------------------------------------------------------------------

    async def search_knowledge(
        self,
        query: str,
        contexts: list[str] | None = None,
        n_results: int = 5,
        min_score: float = 0.7,
    ) -> list[KnowledgeSearchResult]:
        """Semantic search with optional context filtering and score threshold.

        Returns :class:`KnowledgeSearchResult` instances enriched with DB
        metadata from the :class:`KnowledgeItem` table.
        """
        embedding = await self._embedder.embed_single(query)

        # Build vector-store filter for contexts if requested
        vs_filter: dict | None = None
        # Note: Chroma WHERE filters use exact match; for context-list
        # matching we do post-filtering on DB items instead.

        raw_results = await self._vectorstore.search(
            query_embedding=embedding,
            n_results=n_results * 3,  # over-fetch to allow post-filtering
        )

        # Score filter
        raw_results = [r for r in raw_results if r.score >= min_score]

        # Load corresponding KnowledgeItems from DB for enrichment
        enriched: list[KnowledgeSearchResult] = []
        for sr in raw_results:
            # Look up by embedding_id
            stmt = select(KnowledgeItem).where(KnowledgeItem.embedding_id == sr.id)
            result = await self._session.execute(stmt)
            item = result.scalar_one_or_none()
            if item is None:
                continue

            # Context filtering on the DB item
            if contexts:
                item_contexts = item.contexts or []
                if not any(c in item_contexts for c in contexts):
                    continue

            enriched.append(
                KnowledgeSearchResult(
                    item=item,
                    score=sr.score,
                    highlights=[sr.content[:200]],
                )
            )

            if len(enriched) >= n_results:
                break

        # Sort by score descending
        enriched.sort(key=lambda r: r.score, reverse=True)

        logger.info(
            "knowledge_search",
            query_len=len(query),
            contexts=contexts,
            raw_results=len(raw_results),
            returned=len(enriched),
        )
        return enriched

    # ------------------------------------------------------------------
    # 3. extract_knowledge_from_conversation
    # ------------------------------------------------------------------

    async def extract_knowledge_from_conversation(
        self,
        conversation_id: str,
        force: bool = False,
    ) -> list[KnowledgeItem]:
        """Use the LLM to extract knowledge items from a conversation.

        By default, only processes messages that have not yet been extracted
        (``extracted_knowledge == False``).  Set *force=True* to re-extract
        all messages.

        Created items are marked ``verified=False`` (pending admin review).
        """
        from src.llm.base import get_llm_provider

        # Load conversation messages
        stmt = (
            select(Message)
            .where(Message.conversation_id == conversation_id)
            .order_by(Message.created_at.asc())
        )
        if not force:
            stmt = stmt.where(Message.extracted_knowledge.is_(False))
        result = await self._session.execute(stmt)
        messages = list(result.scalars().all())

        if not messages:
            logger.info("extract_no_messages", conversation_id=conversation_id)
            return []

        # Build conversation text for the LLM
        conv_lines = []
        for msg in messages:
            conv_lines.append(f"{msg.role.value}: {msg.content}")
        conversation_text = "\n".join(conv_lines)

        # Collect detected contexts from all messages
        all_contexts: set[str] = set()
        for msg in messages:
            if msg.detected_contexts:
                all_contexts.update(msg.detected_contexts)

        # Call LLM to extract facts
        llm = get_llm_provider()
        prompt_messages = [
            {"role": "system", "content": _EXTRACT_SYSTEM_PROMPT},
            {"role": "user", "content": _EXTRACT_USER_TEMPLATE.format(
                conversation=conversation_text,
            )},
        ]

        try:
            response_text = await llm.chat(prompt_messages, temperature=0.1)
        except Exception as exc:
            logger.error("extract_llm_error", error=str(exc))
            return []

        # Parse the JSON array from LLM response
        extracted_texts = _parse_json_array(response_text)
        if not extracted_texts:
            logger.info(
                "extract_nothing_found",
                conversation_id=conversation_id,
            )
            return []

        # Create KnowledgeItems for each extracted fact
        items: list[KnowledgeItem] = []
        source_msg_id = messages[-1].id if messages else None
        context_list = sorted(all_contexts)

        for text in extracted_texts:
            text = text.strip()
            if not text:
                continue
            item = await self.add_knowledge(
                content=text,
                contexts=context_list,
                source_type="conversation_extract",
                source_message_id=source_msg_id,
                metadata={"conversation_id": conversation_id},
            )
            items.append(item)

        # Mark all processed messages as extracted
        for msg in messages:
            msg.extracted_knowledge = True
        await self._session.flush()

        logger.info(
            "knowledge_extracted",
            conversation_id=conversation_id,
            items_created=len(items),
        )
        return items

    # ------------------------------------------------------------------
    # 4. build_rag_context
    # ------------------------------------------------------------------

    async def build_rag_context(
        self,
        query: str,
        detected_contexts: list[str],
        max_tokens: int = 2000,
    ) -> str:
        """Build a formatted RAG context string for LLM injection.

        Searches for relevant knowledge, formats each item with source
        attribution, and truncates to *max_tokens* (approximated as
        chars / 4).
        """
        results = await self.search_knowledge(
            query=query,
            contexts=detected_contexts if detected_contexts else None,
            n_results=10,
            min_score=0.5,
        )

        if not results:
            return ""

        max_chars = max_tokens * 4  # rough chars-to-tokens estimate
        parts: list[str] = []
        used_chars = 0

        for r in results:
            source_label = r.item.content_type.value
            contexts_str = ", ".join(r.item.contexts or [])
            entry = (
                f"[Source: {source_label}"
                f"{f' | Contexts: {contexts_str}' if contexts_str else ''}"
                f" | Score: {r.score:.2f}]\n"
                f"{r.item.content}"
            )
            entry_len = len(entry) + 4  # +4 for separator
            if used_chars + entry_len > max_chars:
                break
            parts.append(entry)
            used_chars += entry_len

        context_block = "\n\n---\n\n".join(parts)

        logger.info(
            "rag_context_built",
            query_len=len(query),
            items_used=len(parts),
            chars=used_chars,
        )
        return context_block

    # ------------------------------------------------------------------
    # 5. verify_knowledge
    # ------------------------------------------------------------------

    async def verify_knowledge(
        self,
        item_id: str,
        verified: bool,
        verified_by: str,
    ) -> KnowledgeItem:
        """Admin approves or rejects a knowledge item.

        If *verified* is ``False`` (rejected), the item is removed from
        the vector store but kept in the DB for audit.
        """
        stmt = select(KnowledgeItem).where(KnowledgeItem.id == item_id)
        result = await self._session.execute(stmt)
        item = result.scalar_one_or_none()
        if item is None:
            raise NotFoundError("KnowledgeItem", item_id)

        item.verified = verified
        item.created_by = verified_by  # track who verified

        if not verified and item.embedding_id:
            # Rejected → remove from vector store
            try:
                await self._vectorstore.delete([item.embedding_id])
            except Exception as exc:
                logger.warning("verify_delete_vector_error", error=str(exc))
            item.embedding_id = None

        await self._session.flush()

        logger.info(
            "knowledge_verified",
            item_id=item_id,
            verified=verified,
            verified_by=verified_by,
        )
        return item

    # ------------------------------------------------------------------
    # 6. bulk_import
    # ------------------------------------------------------------------

    async def bulk_import(
        self,
        documents: list[BulkDocument],
        contexts: list[str],
        chunk_size: int | None = None,
        chunk_overlap: int | None = None,
    ) -> ImportResult:
        """Batch-import documents: chunk, embed, and store.

        Each document is split into chunks; each chunk becomes a
        :class:`KnowledgeItem`.  Uses configurable chunk parameters
        or falls back to settings.
        """
        settings = get_settings()
        c_size = chunk_size or settings.chunk_size
        c_overlap = chunk_overlap or settings.chunk_overlap

        total = len(documents)
        imported = 0
        errors: list[str] = []

        for idx, doc in enumerate(documents):
            try:
                chunks = chunk_text(doc.content, chunk_size=c_size, overlap=c_overlap)
                if not chunks:
                    errors.append(f"Document {idx}: empty after chunking")
                    continue

                for i, chunk_text_content in enumerate(chunks):
                    # Deduplication is handled inside add_knowledge
                    await self.add_knowledge(
                        content=chunk_text_content,
                        contexts=contexts,
                        source_type="document",
                        metadata={
                            **doc.metadata,
                            "source": doc.source,
                            "chunk_index": i,
                            "total_chunks": len(chunks),
                        },
                    )
                imported += 1

            except Exception as exc:
                errors.append(f"Document {idx}: {exc}")
                logger.warning("bulk_import_error", doc_index=idx, error=str(exc))

        logger.info(
            "bulk_import_complete",
            total=total,
            imported=imported,
            errors=len(errors),
        )
        return ImportResult(total=total, imported=imported, errors=errors)

    # ------------------------------------------------------------------
    # Document CRUD (preserved from previous implementation)
    # ------------------------------------------------------------------

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

        settings = get_settings()
        chunks = chunk_text(content, chunk_size=settings.chunk_size, overlap=settings.chunk_overlap)
        doc.chunk_count = len(chunks)

        if chunks:
            embeddings = await self._embedder.embed(chunks)
            chunk_ids = [f"{doc.id}_{i}" for i in range(len(chunks))]
            chunk_metas = [
                {"title": title, "doc_id": doc.id, "chunk_index": i, **metadata}
                for i in range(len(chunks))
            ]
            await self._vectorstore.add(
                texts=chunks,
                metadatas=chunk_metas,
                ids=chunk_ids,
                embeddings=embeddings,
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
        chunk_ids = [f"{doc.id}_{i}" for i in range(doc.chunk_count)]
        if chunk_ids:
            await self._vectorstore.delete(chunk_ids)
        await self._session.delete(doc)
        logger.info("document_deleted", doc_id=document_id)

    async def search(self, query: str, top_k: int = 5) -> list[SearchResult]:
        """Simple semantic search (vector store level, no DB enrichment)."""
        embedding = await self._embedder.embed_single(query)
        results = await self._vectorstore.search(embedding, n_results=top_k)
        return results


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _parse_json_array(text: str) -> list[str]:
    """Best-effort parse of a JSON string array from LLM output."""
    import json

    text = text.strip()

    # Try direct parse first
    try:
        parsed = json.loads(text)
        if isinstance(parsed, list):
            return [str(item) for item in parsed if item]
    except json.JSONDecodeError:
        pass

    # Try to extract JSON array from markdown code block
    if "```" in text:
        for block in text.split("```"):
            block = block.strip()
            if block.startswith("json"):
                block = block[4:].strip()
            try:
                parsed = json.loads(block)
                if isinstance(parsed, list):
                    return [str(item) for item in parsed if item]
            except json.JSONDecodeError:
                continue

    # Try to find array pattern in text
    start = text.find("[")
    end = text.rfind("]")
    if start != -1 and end != -1 and end > start:
        try:
            parsed = json.loads(text[start : end + 1])
            if isinstance(parsed, list):
                return [str(item) for item in parsed if item]
        except json.JSONDecodeError:
            pass

    return []


# ---------------------------------------------------------------------------
# FastAPI dependency
# ---------------------------------------------------------------------------


def get_knowledge_service(
    session: AsyncSession = Depends(get_db_session),
) -> KnowledgeService:
    """FastAPI ``Depends`` factory for :class:`KnowledgeService`."""
    return KnowledgeService(session)
