"""
LLM provider factory.

Selects the correct provider class based on ``settings.llm_backend``
and returns a cached singleton instance.

Usage::

    from src.llm.factory import get_llm_provider

    provider = get_llm_provider()
    result = await provider.complete(messages)
"""

from __future__ import annotations

from src.config.settings import LLMBackend, Settings, get_settings
from src.llm.base import LLMProvider

_instance: LLMProvider | None = None


def get_llm_provider(settings: Settings | None = None) -> LLMProvider:
    """Return a singleton :class:`LLMProvider` for the active backend.

    Parameters
    ----------
    settings:
        Override settings (useful for testing).  When *None* the global
        settings singleton is used.
    """
    global _instance  # noqa: PLW0603

    if _instance is not None:
        return _instance

    if settings is None:
        settings = get_settings()

    providers: dict[LLMBackend, type] = {
        LLMBackend.OLLAMA: _lazy_ollama,
        LLMBackend.VLLM: _lazy_vllm,
    }

    factory = providers.get(settings.llm_backend)
    if factory is None:
        raise ValueError(f"Unknown LLM backend: {settings.llm_backend}")

    _instance = factory()
    return _instance


def reset_provider() -> None:
    """Reset the cached provider (for testing)."""
    global _instance  # noqa: PLW0603
    _instance = None


# ── Lazy imports (avoid circular / unnecessary loads) ──────────────────────


def _lazy_ollama() -> LLMProvider:
    from src.llm.ollama import OllamaProvider

    return OllamaProvider()


def _lazy_vllm() -> LLMProvider:
    from src.llm.vllm import VLLMProvider

    return VLLMProvider()
