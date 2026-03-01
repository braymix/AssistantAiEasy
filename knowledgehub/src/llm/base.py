"""Abstract LLM provider interface and factory."""

from abc import ABC, abstractmethod

from src.config.settings import LLMBackend, get_settings


class LLMProvider(ABC):
    """Abstract interface for LLM backends."""

    @abstractmethod
    async def generate(self, prompt: str, **kwargs) -> str:
        """Generate a text completion for the given prompt."""

    @abstractmethod
    async def generate_stream(self, prompt: str, **kwargs):
        """Stream a text completion token by token."""

    @abstractmethod
    async def health_check(self) -> bool:
        """Verify the LLM backend is reachable."""


def get_llm_provider() -> LLMProvider:
    """Factory: return the correct LLM provider for the active profile."""
    settings = get_settings()
    if settings.llm_backend == LLMBackend.VLLM:
        from src.llm.vllm import VLLMProvider

        return VLLMProvider()
    from src.llm.ollama import OllamaProvider

    return OllamaProvider()
