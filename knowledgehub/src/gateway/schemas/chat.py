"""
Pydantic v2 schemas 100% compatible with the OpenAI Chat Completions API.

Open WebUI sends requests in this exact format, so the gateway can act as a
transparent proxy by accepting and returning these types.
"""

import time
import uuid
from typing import Literal

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Request
# ---------------------------------------------------------------------------

class ChatMessage(BaseModel):
    role: Literal["system", "user", "assistant"] = "user"
    content: str


class ChatCompletionRequest(BaseModel):
    """OpenAI-compatible POST /v1/chat/completions body."""

    model: str = ""
    messages: list[ChatMessage]
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    top_p: float = Field(default=1.0, ge=0.0, le=1.0)
    max_tokens: int | None = Field(default=None, ge=1)
    stream: bool = False
    stop: list[str] | str | None = None
    presence_penalty: float = Field(default=0.0, ge=-2.0, le=2.0)
    frequency_penalty: float = Field(default=0.0, ge=-2.0, le=2.0)
    user: str | None = None


# ---------------------------------------------------------------------------
# Response – non-streaming
# ---------------------------------------------------------------------------

class ChoiceMessage(BaseModel):
    role: Literal["assistant"] = "assistant"
    content: str


class Choice(BaseModel):
    index: int = 0
    message: ChoiceMessage
    finish_reason: Literal["stop", "length", "content_filter"] | None = "stop"


class UsageInfo(BaseModel):
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0


class ChatCompletionResponse(BaseModel):
    """OpenAI-compatible response for POST /v1/chat/completions."""

    id: str = Field(default_factory=lambda: f"chatcmpl-{uuid.uuid4().hex[:24]}")
    object: Literal["chat.completion"] = "chat.completion"
    created: int = Field(default_factory=lambda: int(time.time()))
    model: str = ""
    choices: list[Choice]
    usage: UsageInfo = Field(default_factory=UsageInfo)


# ---------------------------------------------------------------------------
# Response – streaming (Server-Sent Events)
# ---------------------------------------------------------------------------

class DeltaMessage(BaseModel):
    """Partial message sent inside a streaming chunk."""

    role: Literal["assistant"] | None = None
    content: str | None = None


class StreamChoice(BaseModel):
    index: int = 0
    delta: DeltaMessage
    finish_reason: Literal["stop", "length", "content_filter"] | None = None


class ChatCompletionChunk(BaseModel):
    """Single SSE chunk in a streaming chat completion."""

    id: str = ""
    object: Literal["chat.completion.chunk"] = "chat.completion.chunk"
    created: int = Field(default_factory=lambda: int(time.time()))
    model: str = ""
    choices: list[StreamChoice]


# ---------------------------------------------------------------------------
# Model listing (Open WebUI calls GET /v1/models)
# ---------------------------------------------------------------------------

class ModelInfo(BaseModel):
    id: str
    object: Literal["model"] = "model"
    owned_by: str = "knowledgehub"


class ModelListResponse(BaseModel):
    object: Literal["list"] = "list"
    data: list[ModelInfo]
