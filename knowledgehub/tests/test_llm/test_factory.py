"""Tests for the LLM provider factory."""

from unittest.mock import MagicMock, patch

import pytest

from src.config.settings import LLMBackend
from src.llm.factory import get_llm_provider, reset_provider
from src.llm.ollama import OllamaProvider
from src.llm.vllm import VLLMProvider


@pytest.fixture(autouse=True)
def _reset():
    """Reset the cached provider before each test."""
    reset_provider()
    yield
    reset_provider()


def test_factory_returns_ollama() -> None:
    settings = MagicMock()
    settings.llm_backend = LLMBackend.OLLAMA

    with patch("src.llm.ollama.get_settings", return_value=settings):
        provider = get_llm_provider(settings)

    assert isinstance(provider, OllamaProvider)


def test_factory_returns_vllm() -> None:
    settings = MagicMock()
    settings.llm_backend = LLMBackend.VLLM

    with patch("src.llm.vllm.get_settings", return_value=settings):
        provider = get_llm_provider(settings)

    assert isinstance(provider, VLLMProvider)


def test_factory_caches_singleton() -> None:
    settings = MagicMock()
    settings.llm_backend = LLMBackend.OLLAMA

    with patch("src.llm.ollama.get_settings", return_value=settings):
        p1 = get_llm_provider(settings)
        p2 = get_llm_provider(settings)

    assert p1 is p2


def test_factory_invalid_backend() -> None:
    settings = MagicMock()
    settings.llm_backend = "nonexistent"

    with pytest.raises(ValueError, match="Unknown LLM backend"):
        get_llm_provider(settings)


def test_reset_provider() -> None:
    settings = MagicMock()
    settings.llm_backend = LLMBackend.OLLAMA

    with patch("src.llm.ollama.get_settings", return_value=settings):
        p1 = get_llm_provider(settings)
        reset_provider()
        p2 = get_llm_provider(settings)

    assert p1 is not p2


def test_backward_compat_import() -> None:
    """The old import path still works."""
    from src.llm.base import get_llm_provider as old_import
    from src.llm import get_llm_provider as init_import

    # Both should be callable
    assert callable(old_import)
    assert callable(init_import)
