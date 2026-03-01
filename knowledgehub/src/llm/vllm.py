"""vLLM provider – for full/enterprise profile with GPU acceleration."""

from collections.abc import AsyncGenerator

import httpx

from src.config.logging import get_logger
from src.config.settings import get_settings
from src.llm.base import LLMProvider
from src.shared.exceptions import LLMError

logger = get_logger(__name__)


class VLLMProvider(LLMProvider):
    """LLM provider backed by a vLLM OpenAI-compatible server."""

    def __init__(self):
        settings = get_settings()
        self._base_url = settings.vllm_base_url.rstrip("/")
        self._model = settings.vllm_model
        self._max_tokens = settings.vllm_max_tokens

    async def generate(self, prompt: str, **kwargs) -> str:
        url = f"{self._base_url}/v1/completions"
        payload = {
            "model": self._model,
            "prompt": prompt,
            "max_tokens": kwargs.get("max_tokens", self._max_tokens),
            "temperature": kwargs.get("temperature", 0.7),
            "stream": False,
        }
        try:
            async with httpx.AsyncClient(timeout=120) as client:
                response = await client.post(url, json=payload)
                response.raise_for_status()
                data = response.json()
                choices = data.get("choices", [])
                if choices:
                    return choices[0].get("text", "")
                return ""
        except httpx.HTTPError as exc:
            logger.error("vllm_generate_error", error=str(exc))
            raise LLMError(f"vLLM generation failed: {exc}") from exc

    async def generate_stream(self, prompt: str, **kwargs) -> AsyncGenerator[str, None]:
        url = f"{self._base_url}/v1/completions"
        payload = {
            "model": self._model,
            "prompt": prompt,
            "max_tokens": kwargs.get("max_tokens", self._max_tokens),
            "temperature": kwargs.get("temperature", 0.7),
            "stream": True,
        }
        try:
            async with httpx.AsyncClient(timeout=120) as client:
                async with client.stream("POST", url, json=payload) as response:
                    response.raise_for_status()
                    async for line in response.aiter_lines():
                        if line.startswith("data: "):
                            chunk = line[6:]
                            if chunk.strip() == "[DONE]":
                                break
                            import json

                            data = json.loads(chunk)
                            choices = data.get("choices", [])
                            if choices:
                                token = choices[0].get("text", "")
                                if token:
                                    yield token
        except httpx.HTTPError as exc:
            logger.error("vllm_stream_error", error=str(exc))
            raise LLMError(f"vLLM streaming failed: {exc}") from exc

    async def health_check(self) -> bool:
        try:
            async with httpx.AsyncClient(timeout=5) as client:
                response = await client.get(f"{self._base_url}/v1/models")
                return response.status_code == 200
        except httpx.HTTPError:
            return False
