"""
OpenAI-compatible chat completions proxy.

POST /v1/chat/completions
GET  /v1/models

Flow for every request:
  1. Receive the request from Open WebUI (OpenAI format)
  2. Persist conversation + user message in the DB via ConversationManager
  3. Run the Detection Engine on the latest user message
  4. If relevant contexts are detected → retrieve knowledge via RAG and
     inject a system message with the retrieved context
  5. Forward the (possibly enriched) messages to the LLM backend
  6. Return the response — supports both blocking JSON and streaming SSE
  7. Persist the assistant response via ConversationManager
"""

import time
import uuid
from collections.abc import AsyncGenerator

from fastapi import APIRouter, Depends, Request
from fastapi.responses import StreamingResponse
from sqlalchemy.ext.asyncio import AsyncSession

from src.config import get_settings
from src.config.logging import get_logger
from src.detection.engine import DetectionEngine
from src.gateway.schemas.chat import (
    ChatCompletionChunk,
    ChatCompletionRequest,
    ChatCompletionResponse,
    Choice,
    ChoiceMessage,
    DeltaMessage,
    ModelInfo,
    ModelListResponse,
    StreamChoice,
    UsageInfo,
)
from src.gateway.services.conversation import ConversationManager, get_conversation_manager
from src.knowledge.service import KnowledgeService
from src.llm.base import get_llm_provider
from src.shared.database import get_db_session
from src.shared.exceptions import LLMError

logger = get_logger(__name__)

router = APIRouter()

RAG_SYSTEM_TEMPLATE = (
    "Use the following knowledge base excerpts to inform your answer. "
    "If the excerpts are not relevant, ignore them.\n\n"
    "---BEGIN KNOWLEDGE---\n{context}\n---END KNOWLEDGE---"
)


# ═══════════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════════

def _resolve_model(requested: str) -> str:
    """Return the actual model name from settings if the caller left it blank."""
    if requested:
        return requested
    settings = get_settings()
    if settings.llm_backend.value == "ollama":
        return settings.ollama_model
    return settings.vllm_model


def _extract_last_user_text(messages: list[dict]) -> str:
    """Return the content of the last user message."""
    for msg in reversed(messages):
        if msg.get("role") == "user":
            return msg.get("content", "")
    return ""


async def _detect_and_enrich(
    session: AsyncSession,
    user_text: str,
    messages_dicts: list[dict],
) -> tuple[list[dict], list[str]]:
    """Run detection on user text; if contexts found, inject RAG system message."""
    engine = DetectionEngine(session)
    detection = await engine.detect(user_text)
    detected_topics = detection.suggested_topics

    if detection.confidence < 0.3 or not detected_topics:
        return messages_dicts, detected_topics

    # Retrieve knowledge for the detected contexts
    service = KnowledgeService(session)
    results = await service.search(user_text, top_k=5)

    if not results:
        return messages_dicts, detected_topics

    context_block = "\n\n---\n\n".join(r.content for r in results)
    rag_system = RAG_SYSTEM_TEMPLATE.format(context=context_block)

    logger.info(
        "rag_enrichment",
        topics=detected_topics,
        chunks=len(results),
    )

    # Inject the RAG system message right after any existing system messages
    enriched: list[dict] = []
    injected = False
    for msg in messages_dicts:
        enriched.append(msg)
        if msg["role"] == "system" and not injected:
            enriched.append({"role": "system", "content": rag_system})
            injected = True

    # If there were no system messages, prepend the RAG system message
    if not injected:
        enriched.insert(0, {"role": "system", "content": rag_system})

    return enriched, detected_topics


# ═══════════════════════════════════════════════════════════════════════════
# Routes
# ═══════════════════════════════════════════════════════════════════════════

@router.post("/v1/chat/completions")
async def chat_completions(
    body: ChatCompletionRequest,
    request: Request,
    session: AsyncSession = Depends(get_db_session),
    conv_mgr: ConversationManager = Depends(get_conversation_manager),
):
    """OpenAI-compatible chat completions endpoint (proxy with intelligence)."""
    model = _resolve_model(body.model)
    messages_dicts = [m.model_dump() for m in body.messages]
    user_text = _extract_last_user_text(messages_dicts)

    # 1. Create or retrieve existing conversation via ConversationManager
    session_id = body.user or request.headers.get("X-Request-ID", str(uuid.uuid4()))
    conv = await conv_mgr.create_or_get_conversation(session_id)

    # Persist all incoming messages (only the last user message is tracked for detection)
    last_user_msg = None
    for msg in messages_dicts:
        db_msg = await conv_mgr.add_message(
            conversation_id=conv.id,
            role=msg.get("role", "user"),
            content=msg.get("content", ""),
        )
        if msg.get("role") == "user":
            last_user_msg = db_msg

    # 2. Detect context & RAG enrichment
    enriched_messages, detected_topics = await _detect_and_enrich(
        session, user_text, messages_dicts,
    )

    # Update user message with detected contexts
    if last_user_msg and detected_topics:
        last_user_msg.detected_contexts = detected_topics
        await session.flush()

    # Build kwargs to forward to the LLM
    request_kwargs: dict = {
        "temperature": body.temperature,
    }
    if body.max_tokens is not None:
        request_kwargs["max_tokens"] = body.max_tokens

    completion_id = f"chatcmpl-{uuid.uuid4().hex[:24]}"
    created_ts = int(time.time())

    # ── Streaming mode ─────────────────────────────────────────────────
    if body.stream:
        async def stream_and_persist():
            full_parts: list[str] = []

            llm = get_llm_provider()

            # First chunk: role
            first_chunk = ChatCompletionChunk(
                id=completion_id, created=created_ts, model=model,
                choices=[StreamChoice(delta=DeltaMessage(role="assistant", content=""))],
            )
            yield f"data: {first_chunk.model_dump_json()}\n\n".encode()

            # Content chunks
            try:
                async for token in llm.chat_stream(enriched_messages, **request_kwargs):
                    full_parts.append(token)
                    chunk = ChatCompletionChunk(
                        id=completion_id, created=created_ts, model=model,
                        choices=[StreamChoice(delta=DeltaMessage(content=token))],
                    )
                    yield f"data: {chunk.model_dump_json()}\n\n".encode()
            except LLMError as exc:
                err = ChatCompletionChunk(
                    id=completion_id, created=created_ts, model=model,
                    choices=[StreamChoice(
                        delta=DeltaMessage(content=f"\n\n[Error: {exc.message}]"),
                        finish_reason="stop",
                    )],
                )
                yield f"data: {err.model_dump_json()}\n\n".encode()

            # Stop chunk
            stop = ChatCompletionChunk(
                id=completion_id, created=created_ts, model=model,
                choices=[StreamChoice(delta=DeltaMessage(), finish_reason="stop")],
            )
            yield f"data: {stop.model_dump_json()}\n\n".encode()
            yield b"data: [DONE]\n\n"

            # Persist the complete assistant response via ConversationManager
            full_text = "".join(full_parts)
            if full_text:
                await conv_mgr.add_message(
                    conversation_id=conv.id,
                    role="assistant",
                    content=full_text,
                    metadata={"detected_contexts": detected_topics} if detected_topics else None,
                )
                await session.commit()

        return StreamingResponse(
            stream_and_persist(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",
            },
        )

    # ── Non-streaming mode ─────────────────────────────────────────────
    llm = get_llm_provider()
    try:
        content = await llm.chat(enriched_messages, **request_kwargs)
    except LLMError as exc:
        logger.error("chat_completion_error", error=exc.message)
        content = f"[Error: {exc.message}]"

    # Persist assistant response via ConversationManager
    await conv_mgr.add_message(
        conversation_id=conv.id,
        role="assistant",
        content=content,
        metadata={"detected_contexts": detected_topics} if detected_topics else None,
    )

    return ChatCompletionResponse(
        id=completion_id,
        created=created_ts,
        model=model,
        choices=[Choice(message=ChoiceMessage(content=content))],
        usage=UsageInfo(),
    )


@router.get("/v1/models")
async def list_models():
    """OpenAI-compatible model listing — Open WebUI calls this on startup."""
    settings = get_settings()
    if settings.llm_backend.value == "ollama":
        model_id = settings.ollama_model
    else:
        model_id = settings.vllm_model

    return ModelListResponse(
        data=[ModelInfo(id=model_id, owned_by="knowledgehub")],
    )
