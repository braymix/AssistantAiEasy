"""
Abstract LLM provider interface.

Every concrete backend (Ollama, vLLM, …) must implement this ABC.
The interface provides:

* ``complete`` – unified chat completion (streaming or blocking)
* ``embed``    – text → float vectors
* ``list_models`` / ``health_check`` – introspection & liveness

Backward-compatible convenience wrappers (``chat``, ``chat_stream``,
``generate``, ``generate_stream``) are provided as concrete methods so
existing callers continue to work without changes.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import AsyncGenerator
from typing import Union

from src.llm.models import (
    ChatCompletion,
    ChatCompletionChunk,
    ChatMessage,
    HealthStatus,
    ModelInfo,
)


class LLMProvider(ABC):
    """Abstract interface for LLM backends.

    Concrete providers must implement the four ``@abstractmethod`` s below.
    The legacy helper methods (``chat``, ``chat_stream``, ``generate``,
    ``generate_stream``) delegate to ``complete`` so that old call-sites
    keep working.
    """

    # ── Abstract methods (must be implemented) ─────────────────────────────

    @abstractmethod
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
        """Chat completion – returns a full response or an async generator
        of streaming chunks depending on *stream*.
        """

    @abstractmethod
    async def embed(
        self,
        texts: list[str],
        *,
        model: str | None = None,
    ) -> list[list[float]]:
        """Return embedding vectors for each input text."""

    @abstractmethod
    async def list_models(self) -> list[ModelInfo]:
        """List models available on the backend."""

    @abstractmethod
    async def health_check(self) -> HealthStatus:
        """Check backend connectivity and return structured status."""

    # ── Backward-compatible convenience methods ────────────────────────────

    async def chat(self, messages: list[dict], **kwargs) -> str:
        """Legacy helper: non-streaming chat → plain string.

        Accepts raw dicts (``[{role, content}]``) for compatibility with
        existing gateway and detection code.
        """
        msgs = [ChatMessage(**m) for m in messages]
        result = await self.complete(msgs, stream=False, **kwargs)
        # result is ChatCompletion
        if result.choices:
            return result.choices[0].message.content
        return ""

    async def chat_stream(
        self, messages: list[dict], **kwargs
    ) -> AsyncGenerator[str, None]:
        """Legacy helper: streaming chat → yields content-delta strings."""
        msgs = [ChatMessage(**m) for m in messages]
        gen = await self.complete(msgs, stream=True, **kwargs)
        async for chunk in gen:
            if chunk.choices:
                delta = chunk.choices[0].delta
                if delta.content:
                    yield delta.content

    async def generate(self, prompt: str, **kwargs) -> str:
        """Legacy helper: single-prompt completion → plain string."""
        msgs = [ChatMessage(role="user", content=prompt)]
        result = await self.complete(msgs, stream=False, **kwargs)
        if result.choices:
            return result.choices[0].message.content
        return ""

    async def generate_stream(
        self, prompt: str, **kwargs
    ) -> AsyncGenerator[str, None]:
        """Legacy helper: single-prompt streaming → yields tokens."""
        msgs = [ChatMessage(role="user", content=prompt)]
        gen = await self.complete(msgs, stream=True, **kwargs)
        async for chunk in gen:
            if chunk.choices:
                delta = chunk.choices[0].delta
                if delta.content:
                    yield delta.content


# ── Factory (kept for backward compatibility) ──────────────────────────────

def get_llm_provider() -> LLMProvider:
    """Factory: return the correct LLM provider for the active profile.

    Prefer ``src.llm.factory.get_llm_provider`` for new code – this
    wrapper exists so existing ``from src.llm.base import get_llm_provider``
    call-sites continue to work.
    """
    from src.llm.factory import get_llm_provider as _factory

    return _factory()
