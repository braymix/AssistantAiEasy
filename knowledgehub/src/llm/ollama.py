"""
Ollama LLM provider – for mini profile / local deployment.

Features:
  - Connection pooling via a persistent ``httpx.AsyncClient``
  - Automatic retry with exponential backoff on transient errors
  - Streaming via Ollama's native NDJSON protocol
  - Embedding via ``/api/embed``
  - Model management: pull, info, list, delete
  - Per-request latency / token-rate metrics
"""

from __future__ import annotations

import asyncio
import json as _json
import time
import uuid
from collections.abc import AsyncGenerator
from typing import Any, Callable, Union

import httpx

from src.config.logging import get_logger
from src.config.settings import get_settings
from src.llm.base import LLMProvider
from src.llm.models import (
    ChatCompletion,
    ChatCompletionChunk,
    ChatMessage,
    Choice,
    ChoiceMessage,
    DeltaMessage,
    HealthStatus,
    ModelDetails,
    ModelInfo,
    RequestMetrics,
    StreamChoice,
    UsageInfo,
)
from src.shared.exceptions import LLMError

logger = get_logger(__name__)

# Retry defaults
_MAX_RETRIES = 3
_BACKOFF_BASE = 1.0  # seconds
_BACKOFF_FACTOR = 2.0
_RETRYABLE_STATUS = {502, 503, 504, 429}


class OllamaProvider(LLMProvider):
    """LLM provider backed by a local Ollama instance.

    The provider maintains a persistent ``httpx.AsyncClient`` with connection
    pooling (default: 20 connections) to avoid TCP-handshake overhead per
    request.
    """

    def __init__(self) -> None:
        settings = get_settings()
        self._base_url = settings.ollama_base_url.rstrip("/")
        self._model = settings.ollama_model
        self._timeout = settings.ollama_timeout
        self._client: httpx.AsyncClient | None = None

    # ── HTTP transport (pooled) ────────────────────────────────────────────

    def _get_client(self) -> httpx.AsyncClient:
        """Return or lazily create the persistent connection-pooled client."""
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(
                base_url=self._base_url,
                timeout=httpx.Timeout(self._timeout, connect=10.0),
                limits=httpx.Limits(
                    max_connections=20,
                    max_keepalive_connections=10,
                ),
            )
        return self._client

    async def close(self) -> None:
        """Shutdown the connection pool."""
        if self._client is not None and not self._client.is_closed:
            await self._client.aclose()
            self._client = None

    # ── Retry helper ───────────────────────────────────────────────────────

    async def _retry_request(
        self,
        method: str,
        path: str,
        *,
        json: dict | None = None,
        max_retries: int = _MAX_RETRIES,
    ) -> httpx.Response:
        """Execute an HTTP request with exponential-backoff retry.

        Retries on 502/503/504/429 and connection errors only.
        """
        client = self._get_client()
        last_exc: Exception | None = None

        for attempt in range(max_retries + 1):
            try:
                resp = await client.request(method, path, json=json)

                if resp.status_code in _RETRYABLE_STATUS and attempt < max_retries:
                    wait = _BACKOFF_BASE * (_BACKOFF_FACTOR ** attempt)
                    logger.warning(
                        "ollama_retry",
                        path=path,
                        status=resp.status_code,
                        attempt=attempt + 1,
                        wait=wait,
                    )
                    await asyncio.sleep(wait)
                    continue

                resp.raise_for_status()
                return resp

            except (httpx.ConnectError, httpx.TimeoutException) as exc:
                last_exc = exc
                if attempt < max_retries:
                    wait = _BACKOFF_BASE * (_BACKOFF_FACTOR ** attempt)
                    logger.warning(
                        "ollama_retry_connection",
                        path=path,
                        error=str(exc),
                        attempt=attempt + 1,
                        wait=wait,
                    )
                    await asyncio.sleep(wait)
                    continue
            except httpx.HTTPStatusError as exc:
                raise LLMError(
                    f"Ollama request failed ({exc.response.status_code}): {exc}"
                ) from exc

        raise LLMError(f"Ollama request failed after {max_retries + 1} attempts: {last_exc}")

    # ── complete (core abstract method) ────────────────────────────────────

    async def complete(
        self,
        messages: list[ChatMessage],
        *,
        model: str | None = None,
        temperature: float = 0.7,
        max_tokens: int = 2000,
        stream: bool = False,
        **kwargs,
    ) -> Union[ChatCompletion, AsyncGenerator[ChatCompletionChunk, None]]:
        """Ollama ``/api/chat`` – supports both streaming and non-streaming."""
        effective_model = model or self._model
        msg_dicts = [{"role": m.role, "content": m.content} for m in messages]

        payload: dict[str, Any] = {
            "model": effective_model,
            "messages": msg_dicts,
            "stream": stream,
            "options": {
                "temperature": temperature,
                "num_predict": max_tokens,
                **kwargs,
            },
        }

        if stream:
            return self._stream_chat(payload, effective_model)

        return await self._blocking_chat(payload, effective_model)

    # ── Non-streaming chat ─────────────────────────────────────────────────

    async def _blocking_chat(
        self, payload: dict, model: str
    ) -> ChatCompletion:
        t0 = time.perf_counter()
        try:
            resp = await self._retry_request("POST", "/api/chat", json=payload)
            data = resp.json()
        except LLMError:
            raise
        except Exception as exc:
            raise LLMError(f"Ollama chat failed: {exc}") from exc

        latency = (time.perf_counter() - t0) * 1000
        content = data.get("message", {}).get("content", "")

        # Extract token counts if available
        prompt_tokens = data.get("prompt_eval_count", 0)
        completion_tokens = data.get("eval_count", 0)

        logger.info(
            "ollama_chat_complete",
            model=model,
            latency_ms=round(latency, 1),
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
        )

        completion_id = f"chatcmpl-{uuid.uuid4().hex[:24]}"
        return ChatCompletion(
            id=completion_id,
            model=model,
            choices=[Choice(message=ChoiceMessage(content=content))],
            usage=UsageInfo(
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=prompt_tokens + completion_tokens,
            ),
        )

    # ── Streaming chat ─────────────────────────────────────────────────────

    async def _stream_chat(
        self, payload: dict, model: str
    ) -> AsyncGenerator[ChatCompletionChunk, None]:
        completion_id = f"chatcmpl-{uuid.uuid4().hex[:24]}"
        created = int(time.time())
        t0 = time.perf_counter()
        token_count = 0
        client = self._get_client()

        try:
            async with client.stream("POST", "/api/chat", json=payload) as response:
                response.raise_for_status()

                # First chunk: role
                yield ChatCompletionChunk(
                    id=completion_id,
                    created=created,
                    model=model,
                    choices=[StreamChoice(delta=DeltaMessage(role="assistant", content=""))],
                )

                async for line in response.aiter_lines():
                    if not line:
                        continue
                    data = _json.loads(line)
                    token = data.get("message", {}).get("content", "")

                    if token:
                        token_count += 1
                        yield ChatCompletionChunk(
                            id=completion_id,
                            created=created,
                            model=model,
                            choices=[StreamChoice(delta=DeltaMessage(content=token))],
                        )

                    if data.get("done", False):
                        break

                # Final stop chunk
                yield ChatCompletionChunk(
                    id=completion_id,
                    created=created,
                    model=model,
                    choices=[StreamChoice(
                        delta=DeltaMessage(),
                        finish_reason="stop",
                    )],
                )

        except httpx.HTTPError as exc:
            raise LLMError(f"Ollama chat streaming failed: {exc}") from exc

        latency = (time.perf_counter() - t0) * 1000
        tps = (token_count / (latency / 1000)) if latency > 0 else 0

        logger.info(
            "ollama_stream_complete",
            model=model,
            latency_ms=round(latency, 1),
            tokens=token_count,
            tokens_per_sec=round(tps, 1),
        )

    # ── embed ──────────────────────────────────────────────────────────────

    async def embed(
        self,
        texts: list[str],
        *,
        model: str | None = None,
    ) -> list[list[float]]:
        """Ollama ``/api/embed`` – returns embedding vectors."""
        effective_model = model or self._model
        try:
            resp = await self._retry_request(
                "POST",
                "/api/embed",
                json={"model": effective_model, "input": texts},
            )
            data = resp.json()
            embeddings = data.get("embeddings", [])
            if not embeddings:
                # Older Ollama versions use /api/embeddings (singular input)
                # Fall back to one-at-a-time
                return await self._embed_fallback(texts, effective_model)
            return embeddings
        except LLMError:
            raise
        except Exception as exc:
            raise LLMError(f"Ollama embed failed: {exc}") from exc

    async def _embed_fallback(
        self, texts: list[str], model: str
    ) -> list[list[float]]:
        """Fallback for older Ollama that only supports single-text embed."""
        results: list[list[float]] = []
        for text in texts:
            resp = await self._retry_request(
                "POST",
                "/api/embeddings",
                json={"model": model, "prompt": text},
            )
            data = resp.json()
            results.append(data.get("embedding", []))
        return results

    # ── list_models ────────────────────────────────────────────────────────

    async def list_models(self) -> list[ModelInfo]:
        """Ollama ``/api/tags`` – list locally available models."""
        try:
            resp = await self._retry_request("GET", "/api/tags")
            data = resp.json()
            models = data.get("models", [])
            return [
                ModelInfo(
                    id=m.get("name", ""),
                    owned_by="ollama",
                )
                for m in models
            ]
        except LLMError:
            raise
        except Exception as exc:
            raise LLMError(f"Ollama list models failed: {exc}") from exc

    # ── health_check ───────────────────────────────────────────────────────

    async def health_check(self) -> HealthStatus:
        """Check Ollama connectivity and return structured status."""
        t0 = time.perf_counter()
        try:
            client = self._get_client()
            resp = await client.get("/api/tags")
            latency = (time.perf_counter() - t0) * 1000
            healthy = resp.status_code == 200
            return HealthStatus(
                healthy=healthy,
                backend="ollama",
                latency_ms=round(latency, 1),
                model=self._model,
                detail="" if healthy else f"HTTP {resp.status_code}",
            )
        except Exception as exc:
            latency = (time.perf_counter() - t0) * 1000
            return HealthStatus(
                healthy=False,
                backend="ollama",
                latency_ms=round(latency, 1),
                model=self._model,
                detail=str(exc),
            )

    # ── Ollama-specific methods ────────────────────────────────────────────

    async def pull_model(
        self,
        model: str,
        progress_callback: Callable[[dict], None] | None = None,
    ) -> bool:
        """Pull (download) a model from the Ollama library.

        *progress_callback* is invoked with progress dicts containing
        ``status``, ``total``, ``completed`` fields.

        Returns *True* on success.
        """
        client = self._get_client()
        try:
            async with client.stream(
                "POST",
                "/api/pull",
                json={"name": model, "stream": True},
                timeout=httpx.Timeout(600.0, connect=10.0),
            ) as response:
                response.raise_for_status()
                async for line in response.aiter_lines():
                    if not line:
                        continue
                    data = _json.loads(line)
                    if progress_callback is not None:
                        progress_callback(data)
                    status = data.get("status", "")
                    if status == "success":
                        logger.info("ollama_model_pulled", model=model)
                        return True
            return True
        except Exception as exc:
            logger.error("ollama_pull_failed", model=model, error=str(exc))
            raise LLMError(f"Ollama pull failed for {model}: {exc}") from exc

    async def model_info(self, model: str) -> ModelDetails:
        """Retrieve detailed information about a local model.

        Ollama ``POST /api/show`` returns architecture, parameter count, etc.
        """
        try:
            resp = await self._retry_request(
                "POST", "/api/show", json={"name": model}
            )
            data = resp.json()
            details = data.get("details", {})
            return ModelDetails(
                id=model,
                family=details.get("family", ""),
                parameter_size=details.get("parameter_size", ""),
                quantization=details.get("quantization_level", ""),
                format=details.get("format", ""),
            )
        except LLMError:
            raise
        except Exception as exc:
            raise LLMError(f"Ollama model info failed: {exc}") from exc

    async def generate_with_context(
        self,
        prompt: str,
        context: list[int],
    ) -> str:
        """Generate using Ollama's raw ``/api/generate`` with a pre-existing
        context token list (useful for conversational continuations without
        re-sending the full history).
        """
        try:
            resp = await self._retry_request(
                "POST",
                "/api/generate",
                json={
                    "model": self._model,
                    "prompt": prompt,
                    "context": context,
                    "stream": False,
                },
            )
            data = resp.json()
            return data.get("response", "")
        except LLMError:
            raise
        except Exception as exc:
            raise LLMError(f"Ollama generate_with_context failed: {exc}") from exc

    async def delete_model(self, model: str) -> bool:
        """Delete a locally cached model."""
        try:
            client = self._get_client()
            resp = await client.request("DELETE", "/api/delete", json={"name": model})
            resp.raise_for_status()
            logger.info("ollama_model_deleted", model=model)
            return True
        except Exception as exc:
            logger.error("ollama_delete_failed", model=model, error=str(exc))
            raise LLMError(f"Ollama delete failed for {model}: {exc}") from exc

    async def get_metrics(self) -> RequestMetrics:
        """Return last-request metrics (placeholder – Ollama does not expose
        a global metrics endpoint, so this returns the model name only).
        """
        return RequestMetrics(model=self._model)
