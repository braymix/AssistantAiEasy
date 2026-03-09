"""
Pydantic v2 models for LLM request/response types.

Compatible with the OpenAI Chat Completions format so that both the
Ollama and vLLM providers can consume and produce the same structures.

These models are intentionally decoupled from the gateway-level schemas
in ``src/gateway/schemas/chat.py`` — the gateway schemas model the HTTP
wire format, while these capture the provider-level contract.
"""

from __future__ import annotations

import time
import uuid
from enum import Enum
from typing import Literal

from pydantic import BaseModel, Field


# ═══════════════════════════════════════════════════════════════════════════════
# Chat messages
# ═══════════════════════════════════════════════════════════════════════════════


class ChatMessage(BaseModel):
    """Single message in a conversation."""

    role: Literal["system", "user", "assistant"] = "user"
    content: str = ""


# ═══════════════════════════════════════════════════════════════════════════════
# Completion response (non-streaming)
# ═══════════════════════════════════════════════════════════════════════════════


class UsageInfo(BaseModel):
    """Token usage information."""

    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0


class ChoiceMessage(BaseModel):
    role: Literal["assistant"] = "assistant"
    content: str = ""


class Choice(BaseModel):
    index: int = 0
    message: ChoiceMessage = Field(default_factory=ChoiceMessage)
    finish_reason: Literal["stop", "length", "content_filter"] | None = "stop"


class ChatCompletion(BaseModel):
    """Complete chat completion response (non-streaming)."""

    id: str = Field(default_factory=lambda: f"chatcmpl-{uuid.uuid4().hex[:24]}")
    object: Literal["chat.completion"] = "chat.completion"
    created: int = Field(default_factory=lambda: int(time.time()))
    model: str = ""
    choices: list[Choice] = Field(default_factory=list)
    usage: UsageInfo = Field(default_factory=UsageInfo)


# ═══════════════════════════════════════════════════════════════════════════════
# Streaming chunk
# ═══════════════════════════════════════════════════════════════════════════════


class DeltaMessage(BaseModel):
    """Partial content in a streaming chunk."""

    role: Literal["assistant"] | None = None
    content: str | None = None


class StreamChoice(BaseModel):
    index: int = 0
    delta: DeltaMessage = Field(default_factory=DeltaMessage)
    finish_reason: Literal["stop", "length", "content_filter"] | None = None


class ChatCompletionChunk(BaseModel):
    """Single SSE chunk in a streaming chat completion."""

    id: str = ""
    object: Literal["chat.completion.chunk"] = "chat.completion.chunk"
    created: int = Field(default_factory=lambda: int(time.time()))
    model: str = ""
    choices: list[StreamChoice] = Field(default_factory=list)


# ═══════════════════════════════════════════════════════════════════════════════
# Model info
# ═══════════════════════════════════════════════════════════════════════════════


class ModelInfo(BaseModel):
    """Describes an available model."""

    id: str
    object: Literal["model"] = "model"
    owned_by: str = "knowledgehub"
    created: int = Field(default_factory=lambda: int(time.time()))


class ModelDetails(BaseModel):
    """Extended model metadata (Ollama-specific fields are optional)."""

    id: str
    family: str = ""
    parameter_size: str = ""
    quantization: str = ""
    format: str = ""
    context_length: int = 0


# ═══════════════════════════════════════════════════════════════════════════════
# Health
# ═══════════════════════════════════════════════════════════════════════════════


class HealthStatus(BaseModel):
    """Health check result for an LLM backend."""

    healthy: bool
    backend: str
    latency_ms: float = 0.0
    model: str = ""
    detail: str = ""


# ═══════════════════════════════════════════════════════════════════════════════
# Metrics
# ═══════════════════════════════════════════════════════════════════════════════


class RequestMetrics(BaseModel):
    """Per-request performance metrics."""

    latency_ms: float = 0.0
    tokens_generated: int = 0
    tokens_per_second: float = 0.0
    prompt_tokens: int = 0
    model: str = ""
