"""
title: KnowledgeHub Pipeline
author: KnowledgeHub Team
version: 1.0
description: Integra Open WebUI con KnowledgeHub per RAG aziendale.

Copy this file into Open WebUI's `pipelines/` directory or upload it
through the Admin → Pipelines UI.  The pipeline intercepts every chat
request, forwards it to the KnowledgeHub Gateway (which performs context
detection + RAG enrichment), and streams the response back to the user.

Requirements:
    pip install requests pydantic  (both already available inside Open WebUI)
"""

from __future__ import annotations

import json
import logging
import os
from typing import Generator, Iterator, List, Union

import requests
from pydantic import BaseModel, Field

logger = logging.getLogger("knowledgehub_pipeline")


# ═══════════════════════════════════════════════════════════════════════════════
# Valves – configurable via Open WebUI Admin → Pipelines → KnowledgeHub
# ═══════════════════════════════════════════════════════════════════════════════


class Pipeline:
    """Open WebUI Pipeline that proxies chat requests through KnowledgeHub."""

    class Valves(BaseModel):
        """User-configurable settings exposed in the Open WebUI admin UI."""

        gateway_url: str = Field(
            default="http://localhost:8000",
            description="KnowledgeHub Gateway base URL (e.g. http://knowledgehub-gateway:8000)",
        )
        enable_rag: bool = Field(
            default=True,
            description="Enable RAG enrichment via KnowledgeHub detection engine",
        )
        show_sources: bool = Field(
            default=True,
            description="Append retrieved source references to the response",
        )
        min_confidence: float = Field(
            default=0.7,
            description="Minimum confidence score for knowledge retrieval (0.0 – 1.0)",
        )
        request_timeout: int = Field(
            default=120,
            description="HTTP timeout in seconds for gateway requests",
        )
        fallback_to_direct: bool = Field(
            default=True,
            description="If the gateway is unreachable, fall back to the default Open WebUI model",
        )
        api_key: str = Field(
            default="",
            description="Optional API key sent as X-API-Key header to the gateway",
        )

    # ── Lifecycle ──────────────────────────────────────────────────────────

    def __init__(self) -> None:
        self.name = "KnowledgeHub Pipeline"
        self.valves = self.Valves(
            gateway_url=os.getenv(
                "KNOWLEDGEHUB_GATEWAY_URL", "http://localhost:8000"
            ),
            api_key=os.getenv("KNOWLEDGEHUB_API_KEY", ""),
        )
        self._gateway_healthy: bool = False

    async def on_startup(self) -> None:
        """Called when Open WebUI starts – verify gateway connectivity."""
        logger.info("KnowledgeHub pipeline starting, gateway=%s", self.valves.gateway_url)
        self._gateway_healthy = self._check_gateway_health()
        if self._gateway_healthy:
            logger.info("KnowledgeHub gateway is reachable")
        else:
            logger.warning(
                "KnowledgeHub gateway is NOT reachable at %s – "
                "requests will fall back to the default model if fallback is enabled",
                self.valves.gateway_url,
            )

    async def on_shutdown(self) -> None:
        """Called when Open WebUI shuts down."""
        logger.info("KnowledgeHub pipeline shutting down")

    async def on_valves_updated(self) -> None:
        """Called when the user changes Valves in the admin UI."""
        logger.info("Valves updated, re-checking gateway at %s", self.valves.gateway_url)
        self._gateway_healthy = self._check_gateway_health()

    # ── Health check ───────────────────────────────────────────────────────

    def _check_gateway_health(self) -> bool:
        """Synchronous HTTP health check against the gateway."""
        url = f"{self.valves.gateway_url.rstrip('/')}/health"
        try:
            resp = requests.get(url, timeout=5)
            return resp.status_code == 200
        except requests.RequestException:
            return False

    # ── Main pipe method ───────────────────────────────────────────────────

    def pipe(
        self,
        user_message: str,
        model_id: str,
        messages: List[dict],
        body: dict,
    ) -> Union[str, Generator, Iterator]:
        """
        Process a chat request through KnowledgeHub Gateway.

        Open WebUI calls this method for every user message.  The pipeline:
          1. Builds an OpenAI-compatible request payload
          2. Sends it to the KnowledgeHub Gateway
          3. Returns either a streaming generator or a plain string

        If the gateway is unreachable and ``fallback_to_direct`` is enabled,
        the method returns ``None`` so Open WebUI processes the request
        with its default model.
        """
        # If RAG is disabled, let Open WebUI handle the request normally
        if not self.valves.enable_rag:
            return None

        gateway_url = self.valves.gateway_url.rstrip("/")
        url = f"{gateway_url}/v1/chat/completions"

        # Build OpenAI-compatible payload
        payload = self._build_payload(messages, model_id, body)

        headers = {
            "Content-Type": "application/json",
            "Accept": "text/event-stream" if payload.get("stream") else "application/json",
        }
        if self.valves.api_key:
            headers["X-API-Key"] = self.valves.api_key

        # Add KnowledgeHub-specific headers
        headers["X-KnowledgeHub-MinConfidence"] = str(self.valves.min_confidence)
        headers["X-KnowledgeHub-ShowSources"] = str(self.valves.show_sources).lower()

        try:
            if payload.get("stream"):
                return self._stream_response(url, payload, headers)
            else:
                return self._blocking_response(url, payload, headers)
        except requests.ConnectionError:
            logger.warning("Gateway unreachable at %s", gateway_url)
            self._gateway_healthy = False
            if self.valves.fallback_to_direct:
                logger.info("Falling back to default Open WebUI model")
                return None
            return "[KnowledgeHub] Gateway non raggiungibile. Riprova più tardi."
        except requests.Timeout:
            logger.warning("Gateway request timed out after %ds", self.valves.request_timeout)
            if self.valves.fallback_to_direct:
                return None
            return "[KnowledgeHub] Timeout nella richiesta al gateway."
        except Exception as exc:
            logger.exception("Unexpected error in KnowledgeHub pipeline")
            if self.valves.fallback_to_direct:
                return None
            return f"[KnowledgeHub] Errore: {exc}"

    # ── Payload builder ────────────────────────────────────────────────────

    def _build_payload(
        self, messages: List[dict], model_id: str, body: dict
    ) -> dict:
        """Build the OpenAI-compatible request payload for the gateway."""
        payload: dict = {
            "model": model_id,
            "messages": messages,
            "stream": body.get("stream", True),
        }

        # Forward optional parameters if present
        for key in ("temperature", "top_p", "max_tokens", "stop",
                     "presence_penalty", "frequency_penalty"):
            if key in body and body[key] is not None:
                payload[key] = body[key]

        # Pass user identifier for conversation tracking
        if "user" in body:
            payload["user"] = body["user"]

        return payload

    # ── Streaming response ─────────────────────────────────────────────────

    def _stream_response(
        self, url: str, payload: dict, headers: dict
    ) -> Generator[str, None, None]:
        """
        Send a streaming request to the gateway and yield content tokens.

        Parses SSE lines in the ``data: {...}`` format used by the
        OpenAI streaming protocol.
        """
        resp = requests.post(
            url,
            json=payload,
            headers=headers,
            stream=True,
            timeout=self.valves.request_timeout,
        )

        if resp.status_code != 200:
            error_text = resp.text[:500]
            logger.error(
                "Gateway returned %d: %s", resp.status_code, error_text,
            )
            yield f"[KnowledgeHub] Errore gateway (HTTP {resp.status_code})"
            return

        self._gateway_healthy = True

        for line in resp.iter_lines(decode_unicode=True):
            if not line:
                continue

            # SSE format: "data: {json}" or "data: [DONE]"
            if not line.startswith("data: "):
                continue

            data_str = line[6:]  # strip "data: " prefix

            if data_str.strip() == "[DONE]":
                return

            try:
                chunk = json.loads(data_str)
            except json.JSONDecodeError:
                continue

            # Extract content delta from the OpenAI chunk format
            choices = chunk.get("choices", [])
            if not choices:
                continue

            delta = choices[0].get("delta", {})
            content = delta.get("content")

            if content:
                yield content

    # ── Blocking (non-streaming) response ──────────────────────────────────

    def _blocking_response(
        self, url: str, payload: dict, headers: dict
    ) -> str:
        """Send a non-streaming request and return the full response text."""
        resp = requests.post(
            url,
            json=payload,
            headers=headers,
            timeout=self.valves.request_timeout,
        )

        if resp.status_code != 200:
            error_text = resp.text[:500]
            logger.error(
                "Gateway returned %d: %s", resp.status_code, error_text,
            )
            return f"[KnowledgeHub] Errore gateway (HTTP {resp.status_code})"

        self._gateway_healthy = True

        try:
            data = resp.json()
        except json.JSONDecodeError:
            return resp.text

        # Extract content from OpenAI-format response
        choices = data.get("choices", [])
        if not choices:
            return "[KnowledgeHub] Risposta vuota dal gateway"

        message = choices[0].get("message", {})
        return message.get("content", "")
