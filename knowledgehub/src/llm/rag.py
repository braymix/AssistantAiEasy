"""
RAG Orchestrator – end-to-end Retrieval-Augmented Generation.

Coordinates the LLM provider, Knowledge Service, and prompt templates
to deliver context-enriched chat completions.  Designed to be the
single entry point for RAG-powered conversations.

Flow::

    user message
         │
         ▼
    ┌─────────────┐
    │ extract query│
    └──────┬──────┘
           ▼
    ┌──────────────────┐
    │ search knowledge │──▶ vector store + context filter
    └──────┬───────────┘
           ▼
    ┌──────────────────┐
    │ (opt.) rerank    │──▶ LLM-based reranking
    └──────┬───────────┘
           ▼
    ┌──────────────────┐
    │ build RAG prompt │──▶ inject system message
    └──────┬───────────┘
           ▼
    ┌──────────────────┐
    │ LLM complete     │──▶ streaming or blocking
    └──────┬───────────┘
           ▼
    ┌──────────────────┐
    │ post-process     │──▶ source attribution
    └──────────────────┘

Usage::

    rag = RAGOrchestrator(llm, knowledge_svc, settings)
    result = await rag.generate_with_rag(messages, ["IT", "HR"])
"""

from __future__ import annotations

import json
import time
from collections import OrderedDict
from collections.abc import AsyncGenerator
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Union

from src.config.logging import get_logger
from src.config.settings import Settings, get_settings
from src.llm.models import (
    ChatCompletion,
    ChatCompletionChunk,
    ChatMessage,
    Choice,
    ChoiceMessage,
    DeltaMessage,
    StreamChoice,
)
from src.llm.prompts import (
    KNOWLEDGE_EXTRACTION_PROMPT,
    KNOWLEDGE_EXTRACTION_USER_TEMPLATE,
    RAG_SYSTEM_PROMPT,
    RERANK_PROMPT,
    SOURCE_ATTRIBUTION_HEADER,
    SOURCE_ATTRIBUTION_ITEM,
    estimate_tokens,
    truncate_to_tokens,
)
from src.shared.exceptions import LLMError

if TYPE_CHECKING:
    from src.knowledge.service import KnowledgeSearchResult, KnowledgeService
    from src.llm.base import LLMProvider
    from src.shared.models import Conversation, KnowledgeItem

logger = get_logger(__name__)


# ═══════════════════════════════════════════════════════════════════════════════
# Result containers
# ═══════════════════════════════════════════════════════════════════════════════


@dataclass
class RAGResult:
    """Full result from a RAG-enriched generation."""

    completion: ChatCompletion | None = None
    sources_used: list[dict] = field(default_factory=list)
    knowledge_count: int = 0
    rag_tokens: int = 0
    fallback: bool = False  # True if RAG failed and we used plain LLM


# ═══════════════════════════════════════════════════════════════════════════════
# Query cache (LRU + TTL)
# ═══════════════════════════════════════════════════════════════════════════════


class _QueryCache:
    """Simple LRU cache for RAG prompt results.  Avoids re-searching
    the vector store for repeated identical queries within the TTL window.
    """

    def __init__(self, maxsize: int = 128, ttl: float = 300.0) -> None:
        self._maxsize = maxsize
        self._ttl = ttl
        self._store: OrderedDict[str, tuple[float, Any]] = OrderedDict()

    def get(self, key: str) -> Any | None:
        entry = self._store.get(key)
        if entry is None:
            return None
        ts, value = entry
        if (time.monotonic() - ts) > self._ttl:
            self._store.pop(key, None)
            return None
        self._store.move_to_end(key)
        return value

    def put(self, key: str, value: Any) -> None:
        self._store[key] = (time.monotonic(), value)
        self._store.move_to_end(key)
        while len(self._store) > self._maxsize:
            self._store.popitem(last=False)

    def clear(self) -> None:
        self._store.clear()


# ═══════════════════════════════════════════════════════════════════════════════
# RAG Orchestrator
# ═══════════════════════════════════════════════════════════════════════════════


class RAGOrchestrator:
    """End-to-end Retrieval-Augmented Generation orchestrator.

    Parameters
    ----------
    llm_provider:
        The LLM backend used for generation and (optionally) reranking.
    knowledge_service:
        The knowledge base search and storage service.
    settings:
        Application settings.  If *None*, the global singleton is used.
    rag_system_prompt:
        Override the default RAG system prompt template.
    show_sources:
        Whether to append source attribution to non-streaming responses.
    enable_rerank:
        Whether to use LLM-based reranking of search results.
    cache_ttl:
        TTL (seconds) for the query-level RAG cache.
    """

    def __init__(
        self,
        llm_provider: LLMProvider,
        knowledge_service: KnowledgeService,
        settings: Settings | None = None,
        *,
        rag_system_prompt: str | None = None,
        show_sources: bool = True,
        enable_rerank: bool = False,
        cache_ttl: float = 300.0,
    ) -> None:
        self._llm = llm_provider
        self._knowledge = knowledge_service
        self._settings = settings or get_settings()
        self._rag_prompt = rag_system_prompt or RAG_SYSTEM_PROMPT
        self._show_sources = show_sources
        self._enable_rerank = enable_rerank
        self._cache = _QueryCache(ttl=cache_ttl)

    # ── 1. generate_with_rag ───────────────────────────────────────────────

    async def generate_with_rag(
        self,
        messages: list[ChatMessage],
        detected_contexts: list[str],
        *,
        model: str | None = None,
        stream: bool = True,
        temperature: float = 0.7,
        max_tokens: int = 2000,
        max_context_tokens: int = 2000,
        min_score: float = 0.5,
        **kwargs,
    ) -> Union[ChatCompletion, AsyncGenerator[ChatCompletionChunk, None]]:
        """Generate a RAG-enriched response.

        Flow:
          a) Extract query from the last user message
          b) Search knowledge base for relevant results
          c) (Optional) Rerank results with LLM
          d) Build system prompt with RAG context
          e) Generate response with LLM
          f) Post-process (source attribution if configured)
          g) Return response — falls back to plain LLM if RAG fails
        """
        # a) Extract query
        query = self._extract_query(messages)

        # b+c+d) Build enriched messages (with fallback)
        enriched_messages, sources = await self._build_enriched_messages(
            messages=messages,
            query=query,
            contexts=detected_contexts,
            max_context_tokens=max_context_tokens,
            min_score=min_score,
        )

        # e) Generate with LLM
        try:
            result = await self._llm.complete(
                enriched_messages,
                model=model,
                temperature=temperature,
                max_tokens=max_tokens,
                stream=stream,
                **kwargs,
            )
        except LLMError:
            raise
        except Exception as exc:
            raise LLMError(f"RAG generation failed: {exc}") from exc

        # f) Post-process
        if stream:
            return self._post_process_stream(result, sources)

        return self._post_process_blocking(result, sources)

    # ── 2. build_rag_prompt ────────────────────────────────────────────────

    async def build_rag_prompt(
        self,
        query: str,
        knowledge_results: list[KnowledgeSearchResult],
        max_context_tokens: int = 2000,
    ) -> str:
        """Assemble the RAG system prompt from search results.

        Formats each knowledge result with source attribution and truncates
        to the token budget.
        """
        if not knowledge_results:
            return ""

        parts: list[str] = []
        used_tokens = 0
        # Reserve tokens for the prompt template chrome
        template_tokens = estimate_tokens(
            self._rag_prompt.replace("{knowledge}", "")
        )
        available_tokens = max_context_tokens - template_tokens

        for i, r in enumerate(knowledge_results, 1):
            source_label = r.item.content_type.value if hasattr(r.item, "content_type") else "unknown"
            contexts_str = ", ".join(r.item.contexts or []) if hasattr(r.item, "contexts") else ""

            entry = (
                f"[{i}] [Fonte: {source_label}"
                f"{f' | Contesto: {contexts_str}' if contexts_str else ''}"
                f" | Rilevanza: {r.score:.0%}]\n"
                f"{r.item.content}"
            )

            entry_tokens = estimate_tokens(entry) + 2  # +2 for separator
            if used_tokens + entry_tokens > available_tokens:
                # Try truncating this last entry to fit
                remaining = available_tokens - used_tokens - 2
                if remaining > 50:  # only include if meaningful
                    entry = truncate_to_tokens(entry, remaining)
                    parts.append(entry)
                break

            parts.append(entry)
            used_tokens += entry_tokens

        if not parts:
            return ""

        knowledge_block = "\n\n---\n\n".join(parts)
        return self._rag_prompt.format(knowledge=knowledge_block)

    # ── 3. extract_and_store_knowledge ─────────────────────────────────────

    async def extract_and_store_knowledge(
        self,
        conversation: Conversation,
        assistant_response: str,
        detected_contexts: list[str],
        *,
        user_message: str = "",
        min_confidence: float = 0.6,
    ) -> list[KnowledgeItem]:
        """Use the LLM to extract new knowledge from an assistant response.

        Extracted items are stored as *unverified* knowledge items in the
        knowledge base.

        Returns the list of newly created items.
        """
        if not assistant_response.strip():
            return []

        contexts_str = ", ".join(detected_contexts) if detected_contexts else "general"

        extraction_prompt = KNOWLEDGE_EXTRACTION_USER_TEMPLATE.format(
            user_message=user_message,
            assistant_response=assistant_response,
            contexts=contexts_str,
        )

        try:
            extraction_result = await self._llm.chat([
                {"role": "system", "content": KNOWLEDGE_EXTRACTION_PROMPT},
                {"role": "user", "content": extraction_prompt},
            ])
        except Exception as exc:
            logger.warning("knowledge_extraction_llm_failed", error=str(exc))
            return []

        # Parse the LLM response as JSON
        items = self._parse_extraction_response(extraction_result, min_confidence)

        if not items:
            logger.debug("knowledge_extraction_empty", contexts=contexts_str)
            return []

        # Store each extracted item
        created: list[KnowledgeItem] = []
        source_msg_id = None
        if hasattr(conversation, "messages") and conversation.messages:
            # Use the last assistant message id if available
            for msg in reversed(conversation.messages):
                if msg.role.value == "assistant":
                    source_msg_id = str(msg.id)
                    break

        for item_data in items:
            try:
                ki = await self._knowledge.add_knowledge(
                    content=item_data["content"],
                    contexts=detected_contexts,
                    source_type="conversation_extract",
                    source_message_id=source_msg_id,
                )
                created.append(ki)
            except Exception as exc:
                logger.warning(
                    "knowledge_store_failed",
                    content_preview=item_data["content"][:80],
                    error=str(exc),
                )

        logger.info(
            "knowledge_extracted",
            conversation_id=str(conversation.id) if hasattr(conversation, "id") else "unknown",
            items_extracted=len(items),
            items_stored=len(created),
        )
        return created

    # ── 4. rerank_results ──────────────────────────────────────────────────

    async def rerank_results(
        self,
        query: str,
        results: list[KnowledgeSearchResult],
        top_k: int = 5,
    ) -> list[KnowledgeSearchResult]:
        """Use the LLM to rerank search results by relevance.

        Falls back to the original ordering if the LLM call fails.
        """
        if not results or len(results) <= 1:
            return results[:top_k]

        # Build excerpts for the reranking prompt
        excerpts_parts = []
        for i, r in enumerate(results):
            preview = r.item.content[:300] if len(r.item.content) > 300 else r.item.content
            excerpts_parts.append(f"[{i}] {preview}")
        excerpts = "\n\n".join(excerpts_parts)

        prompt = RERANK_PROMPT.format(query=query, excerpts=excerpts)

        try:
            response = await self._llm.chat([
                {"role": "user", "content": prompt},
            ])

            indices = self._parse_rerank_response(response, len(results))
            if indices:
                reranked = [results[i] for i in indices if i < len(results)]
                logger.info(
                    "rerank_complete",
                    original_count=len(results),
                    reranked_count=len(reranked),
                )
                return reranked[:top_k]
        except Exception as exc:
            logger.warning("rerank_failed", error=str(exc))

        # Fallback: keep original ordering
        return results[:top_k]

    # ── Internal helpers ───────────────────────────────────────────────────

    def _extract_query(self, messages: list[ChatMessage]) -> str:
        """Extract the search query from the last user message."""
        for msg in reversed(messages):
            if msg.role == "user" and msg.content.strip():
                return msg.content.strip()
        return ""

    async def _build_enriched_messages(
        self,
        messages: list[ChatMessage],
        query: str,
        contexts: list[str],
        max_context_tokens: int,
        min_score: float,
    ) -> tuple[list[ChatMessage], list[dict]]:
        """Search knowledge, optionally rerank, build prompt, and prepend
        a system message.  Returns (enriched_messages, sources_metadata).

        Falls back to the original messages if RAG fails.
        """
        sources: list[dict] = []

        if not query:
            return messages, sources

        # Check cache
        cache_key = f"{query}|{'|'.join(sorted(contexts))}"
        cached = self._cache.get(cache_key)
        if cached is not None:
            rag_prompt, sources = cached
            logger.debug("rag_cache_hit", query_len=len(query))
        else:
            # Search knowledge base
            try:
                results = await self._knowledge.search_knowledge(
                    query=query,
                    contexts=contexts if contexts else None,
                    n_results=10,
                    min_score=min_score,
                )
            except Exception as exc:
                logger.warning("rag_search_failed", error=str(exc))
                return messages, sources

            if not results:
                return messages, sources

            # Optional reranking
            if self._enable_rerank and len(results) > 3:
                results = await self.rerank_results(query, results, top_k=5)

            # Build RAG prompt
            rag_prompt = await self.build_rag_prompt(
                query=query,
                knowledge_results=results,
                max_context_tokens=max_context_tokens,
            )

            # Collect source metadata for attribution
            sources = [
                {
                    "source_type": r.item.content_type.value if hasattr(r.item, "content_type") else "unknown",
                    "contexts": r.item.contexts or [] if hasattr(r.item, "contexts") else [],
                    "score": r.score,
                    "content_preview": r.item.content[:100],
                }
                for r in results
            ]

            if rag_prompt:
                self._cache.put(cache_key, (rag_prompt, sources))

        if not rag_prompt:
            return messages, sources

        # Prepend RAG system message
        rag_msg = ChatMessage(role="system", content=rag_prompt)
        enriched = [rag_msg] + list(messages)

        logger.info(
            "rag_enriched",
            query_len=len(query),
            sources_count=len(sources),
            rag_tokens=estimate_tokens(rag_prompt),
        )
        return enriched, sources

    def _post_process_blocking(
        self,
        completion: ChatCompletion,
        sources: list[dict],
    ) -> ChatCompletion:
        """Append source attribution to a non-streaming completion."""
        if not self._show_sources or not sources or not completion.choices:
            return completion

        content = completion.choices[0].message.content
        attribution = self._format_sources(sources)
        new_content = content + attribution

        completion.choices[0] = Choice(
            index=0,
            message=ChoiceMessage(content=new_content),
            finish_reason=completion.choices[0].finish_reason,
        )
        return completion

    async def _post_process_stream(
        self,
        gen: AsyncGenerator[ChatCompletionChunk, None],
        sources: list[dict],
    ) -> AsyncGenerator[ChatCompletionChunk, None]:
        """Yield all chunks, then append source attribution as a final chunk."""
        last_chunk: ChatCompletionChunk | None = None

        async for chunk in gen:
            last_chunk = chunk
            yield chunk

        # After stream ends, append sources
        if self._show_sources and sources and last_chunk is not None:
            attribution = self._format_sources(sources)
            yield ChatCompletionChunk(
                id=last_chunk.id,
                created=last_chunk.created,
                model=last_chunk.model,
                choices=[StreamChoice(
                    delta=DeltaMessage(content=attribution),
                )],
            )

    def _format_sources(self, sources: list[dict]) -> str:
        """Format source attribution as a markdown block."""
        if not sources:
            return ""

        lines = [SOURCE_ATTRIBUTION_HEADER]
        seen = set()
        for s in sources:
            ctx = ", ".join(s.get("contexts", []))
            key = f"{s.get('source_type', '')}|{ctx}"
            if key in seen:
                continue
            seen.add(key)
            lines.append(SOURCE_ATTRIBUTION_ITEM.format(
                source_type=s.get("source_type", "unknown"),
                contexts=ctx or "general",
                score=s.get("score", 0.0),
            ))
        return "".join(lines)

    def _parse_extraction_response(
        self,
        response: str,
        min_confidence: float,
    ) -> list[dict]:
        """Parse the LLM's knowledge extraction response.

        Handles both the structured format (list of {content, confidence})
        and the simple format (list of strings).
        """
        text = response.strip()

        # Find JSON array in the response
        start = text.find("[")
        end = text.rfind("]")
        if start == -1 or end == -1:
            return []

        try:
            data = json.loads(text[start:end + 1])
        except json.JSONDecodeError:
            logger.warning("extraction_json_parse_failed", response_preview=text[:200])
            return []

        if not isinstance(data, list):
            return []

        items: list[dict] = []
        for entry in data:
            if isinstance(entry, str):
                # Simple format: ["fact 1", "fact 2"]
                if entry.strip():
                    items.append({"content": entry.strip(), "confidence": 0.7})
            elif isinstance(entry, dict):
                # Structured format: [{"content": "...", "confidence": 0.9}]
                content = entry.get("content", "").strip()
                confidence = float(entry.get("confidence", 0.7))
                if content and confidence >= min_confidence:
                    items.append({"content": content, "confidence": confidence})

        return items

    @staticmethod
    def _parse_rerank_response(response: str, max_index: int) -> list[int]:
        """Parse the LLM's reranking response as a list of valid indices."""
        text = response.strip()
        start = text.find("[")
        end = text.rfind("]")
        if start == -1 or end == -1:
            return []

        try:
            data = json.loads(text[start:end + 1])
        except json.JSONDecodeError:
            return []

        if not isinstance(data, list):
            return []

        indices = []
        for item in data:
            if isinstance(item, (int, float)):
                idx = int(item)
                if 0 <= idx < max_index and idx not in indices:
                    indices.append(idx)

        return indices

    def clear_cache(self) -> None:
        """Clear the query-level RAG cache."""
        self._cache.clear()
        logger.info("rag_cache_cleared")
