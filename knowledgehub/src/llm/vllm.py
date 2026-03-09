"""
vLLM provider – for full/enterprise profile with GPU acceleration.

vLLM exposes an OpenAI-compatible API, so we use the standard
``/v1/chat/completions``, ``/v1/completions``, ``/v1/embeddings``,
and ``/v1/models`` endpoints.

Features:
  - Connection pooling via persistent ``httpx.AsyncClient``
  - Retry with exponential backoff
  - Batch embedding support
  - LoRA adapter selection via ``model`` parameter
  - Streaming SSE parsing
  - Per-request latency / token-rate metrics
"""

from __future__ import annotations

import asyncio
import json as _json
import time
import uuid
from collections.abc import AsyncGenerator
from typing import Any, Union

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
    ModelInfo,
    RequestMetrics,
    StreamChoice,
    UsageInfo,
)
from src.shared.exceptions import LLMError

logger = get_logger(__name__)

_MAX_RETRIES = 3
_BACKOFF_BASE = 1.0
_BACKOFF_FACTOR = 2.0
_RETRYABLE_STATUS = {502, 503, 504, 429}


class VLLMProvider(LLMProvider):
    """LLM provider backed by a vLLM OpenAI-compatible server.

    Supports both the base model and LoRA adapters – pass the adapter
    name as ``model`` to ``complete()`` to select a fine-tuned variant.
    """

    def __init__(self) -> None:
        settings = get_settings()
        self._base_url = settings.vllm_base_url.rstrip("/")
        self._model = settings.vllm_model
        self._max_tokens = settings.vllm_max_tokens
        self._client: httpx.AsyncClient | None = None

    # ── HTTP transport (pooled) ────────────────────────────────────────────

    def _get_client(self) -> httpx.AsyncClient:
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(
                base_url=self._base_url,
                timeout=httpx.Timeout(120.0, connect=10.0),
                limits=httpx.Limits(
                    max_connections=40,
                    max_keepalive_connections=20,
                ),
            )
        return self._client

    async def close(self) -> None:
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
        client = self._get_client()
        last_exc: Exception | None = None

        for attempt in range(max_retries + 1):
            try:
                resp = await client.request(method, path, json=json)

                if resp.status_code in _RETRYABLE_STATUS and attempt < max_retries:
                    wait = _BACKOFF_BASE * (_BACKOFF_FACTOR ** attempt)
                    logger.warning(
                        "vllm_retry",
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
                        "vllm_retry_connection",
                        path=path,
                        error=str(exc),
                        attempt=attempt + 1,
                        wait=wait,
                    )
                    await asyncio.sleep(wait)
                    continue
            except httpx.HTTPStatusError as exc:
                raise LLMError(
                    f"vLLM request failed ({exc.response.status_code}): {exc}"
                ) from exc

        raise LLMError(f"vLLM request failed after {max_retries + 1} attempts: {last_exc}")

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
        """vLLM ``/v1/chat/completions`` – OpenAI-compatible.

        Pass a LoRA adapter name as *model* to use a fine-tuned variant.
        """
        effective_model = model or self._model
        msg_dicts = [{"role": m.role, "content": m.content} for m in messages]

        payload: dict[str, Any] = {
            "model": effective_model,
            "messages": msg_dicts,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": stream,
        }

        # Forward optional OpenAI-compatible params
        for key in ("top_p", "stop", "presence_penalty", "frequency_penalty",
                     "logprobs", "top_logprobs", "seed"):
            if key in kwargs and kwargs[key] is not None:
                payload[key] = kwargs[key]

        if stream:
            return self._stream_chat(payload, effective_model)

        return await self._blocking_chat(payload, effective_model)

    # ── Non-streaming chat ─────────────────────────────────────────────────

    async def _blocking_chat(
        self, payload: dict, model: str
    ) -> ChatCompletion:
        t0 = time.perf_counter()
        try:
            resp = await self._retry_request("POST", "/v1/chat/completions", json=payload)
            data = resp.json()
        except LLMError:
            raise
        except Exception as exc:
            raise LLMError(f"vLLM chat failed: {exc}") from exc

        latency = (time.perf_counter() - t0) * 1000

        choices_raw = data.get("choices", [])
        choices = []
        for c in choices_raw:
            msg = c.get("message", {})
            choices.append(Choice(
                index=c.get("index", 0),
                message=ChoiceMessage(content=msg.get("content", "")),
                finish_reason=c.get("finish_reason", "stop"),
            ))

        usage_raw = data.get("usage", {})
        usage = UsageInfo(
            prompt_tokens=usage_raw.get("prompt_tokens", 0),
            completion_tokens=usage_raw.get("completion_tokens", 0),
            total_tokens=usage_raw.get("total_tokens", 0),
        )

        logger.info(
            "vllm_chat_complete",
            model=model,
            latency_ms=round(latency, 1),
            prompt_tokens=usage.prompt_tokens,
            completion_tokens=usage.completion_tokens,
        )

        return ChatCompletion(
            id=data.get("id", f"chatcmpl-{uuid.uuid4().hex[:24]}"),
            model=model,
            choices=choices,
            usage=usage,
        )

    # ── Streaming chat ─────────────────────────────────────────────────────

    async def _stream_chat(
        self, payload: dict, model: str
    ) -> AsyncGenerator[ChatCompletionChunk, None]:
        client = self._get_client()
        t0 = time.perf_counter()
        token_count = 0

        try:
            async with client.stream(
                "POST", "/v1/chat/completions", json=payload
            ) as response:
                response.raise_for_status()

                async for line in response.aiter_lines():
                    if not line.startswith("data: "):
                        continue

                    data_str = line[6:]
                    if data_str.strip() == "[DONE]":
                        break

                    data = _json.loads(data_str)
                    choices_raw = data.get("choices", [])
                    if not choices_raw:
                        continue

                    raw_choice = choices_raw[0]
                    delta_raw = raw_choice.get("delta", {})
                    finish = raw_choice.get("finish_reason")

                    chunk = ChatCompletionChunk(
                        id=data.get("id", ""),
                        created=data.get("created", int(time.time())),
                        model=model,
                        choices=[StreamChoice(
                            delta=DeltaMessage(
                                role=delta_raw.get("role"),
                                content=delta_raw.get("content"),
                            ),
                            finish_reason=finish,
                        )],
                    )

                    if delta_raw.get("content"):
                        token_count += 1

                    yield chunk

        except httpx.HTTPError as exc:
            raise LLMError(f"vLLM chat streaming failed: {exc}") from exc

        latency = (time.perf_counter() - t0) * 1000
        tps = (token_count / (latency / 1000)) if latency > 0 else 0

        logger.info(
            "vllm_stream_complete",
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
        """vLLM ``/v1/embeddings`` – batch embedding support."""
        effective_model = model or self._model
        try:
            resp = await self._retry_request(
                "POST",
                "/v1/embeddings",
                json={"model": effective_model, "input": texts},
            )
            data = resp.json()
            # Sort by index to ensure consistent ordering
            entries = sorted(data.get("data", []), key=lambda d: d.get("index", 0))
            return [e.get("embedding", []) for e in entries]
        except LLMError:
            raise
        except Exception as exc:
            raise LLMError(f"vLLM embed failed: {exc}") from exc

    async def embed_batch(
        self,
        texts: list[str],
        *,
        model: str | None = None,
        batch_size: int = 64,
    ) -> list[list[float]]:
        """Batch-embed large lists in chunks to respect server limits."""
        results: list[list[float]] = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            embeddings = await self.embed(batch, model=model)
            results.extend(embeddings)
        return results

    # ── list_models ────────────────────────────────────────────────────────

    async def list_models(self) -> list[ModelInfo]:
        """vLLM ``/v1/models`` – list available models (including LoRA adapters)."""
        try:
            resp = await self._retry_request("GET", "/v1/models")
            data = resp.json()
            models = data.get("data", [])
            return [
                ModelInfo(
                    id=m.get("id", ""),
                    owned_by=m.get("owned_by", "vllm"),
                )
                for m in models
            ]
        except LLMError:
            raise
        except Exception as exc:
            raise LLMError(f"vLLM list models failed: {exc}") from exc

    # ── health_check ───────────────────────────────────────────────────────

    async def health_check(self) -> HealthStatus:
        t0 = time.perf_counter()
        try:
            client = self._get_client()
            resp = await client.get("/v1/models")
            latency = (time.perf_counter() - t0) * 1000
            healthy = resp.status_code == 200
            return HealthStatus(
                healthy=healthy,
                backend="vllm",
                latency_ms=round(latency, 1),
                model=self._model,
                detail="" if healthy else f"HTTP {resp.status_code}",
            )
        except Exception as exc:
            latency = (time.perf_counter() - t0) * 1000
            return HealthStatus(
                healthy=False,
                backend="vllm",
                latency_ms=round(latency, 1),
                model=self._model,
                detail=str(exc),
            )

    # ── vLLM-specific helpers ──────────────────────────────────────────────

    async def get_metrics(self) -> RequestMetrics:
        """Placeholder for vLLM Prometheus metrics scraping."""
        return RequestMetrics(model=self._model)
