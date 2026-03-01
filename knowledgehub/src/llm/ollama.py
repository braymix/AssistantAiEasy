"""Ollama LLM provider – for mini profile / local deployment."""

from collections.abc import AsyncGenerator

import httpx

from src.config.logging import get_logger
from src.config.settings import get_settings
from src.llm.base import LLMProvider
from src.shared.exceptions import LLMError

logger = get_logger(__name__)


class OllamaProvider(LLMProvider):
    """LLM provider backed by a local Ollama instance."""

    def __init__(self):
        settings = get_settings()
        self._base_url = settings.ollama_base_url.rstrip("/")
        self._model = settings.ollama_model
        self._timeout = settings.ollama_timeout

    async def generate(self, prompt: str, **kwargs) -> str:
        url = f"{self._base_url}/api/generate"
        payload = {
            "model": self._model,
            "prompt": prompt,
            "stream": False,
            **kwargs,
        }
        try:
            async with httpx.AsyncClient(timeout=self._timeout) as client:
                response = await client.post(url, json=payload)
                response.raise_for_status()
                data = response.json()
                return data.get("response", "")
        except httpx.HTTPError as exc:
            logger.error("ollama_generate_error", error=str(exc))
            raise LLMError(f"Ollama generation failed: {exc}") from exc

    async def generate_stream(self, prompt: str, **kwargs) -> AsyncGenerator[str, None]:
        url = f"{self._base_url}/api/generate"
        payload = {
            "model": self._model,
            "prompt": prompt,
            "stream": True,
            **kwargs,
        }
        try:
            async with httpx.AsyncClient(timeout=self._timeout) as client:
                async with client.stream("POST", url, json=payload) as response:
                    response.raise_for_status()
                    async for line in response.aiter_lines():
                        if line:
                            import json

                            data = json.loads(line)
                            token = data.get("response", "")
                            if token:
                                yield token
                            if data.get("done", False):
                                break
        except httpx.HTTPError as exc:
            logger.error("ollama_stream_error", error=str(exc))
            raise LLMError(f"Ollama streaming failed: {exc}") from exc

    async def health_check(self) -> bool:
        try:
            async with httpx.AsyncClient(timeout=5) as client:
                response = await client.get(f"{self._base_url}/api/tags")
                return response.status_code == 200
        except httpx.HTTPError:
            return False
