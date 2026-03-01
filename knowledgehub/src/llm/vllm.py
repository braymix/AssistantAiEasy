"""vLLM provider – for full/enterprise profile with GPU acceleration.

vLLM exposes an OpenAI-compatible API so we hit /v1/chat/completions
for chat and /v1/completions for raw prompt generation.
"""

import json as _json
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

    # -- raw prompt ----------------------------------------------------------

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
                choices = response.json().get("choices", [])
                return choices[0].get("text", "") if choices else ""
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
                            data = _json.loads(chunk)
                            choices = data.get("choices", [])
                            if choices:
                                token = choices[0].get("text", "")
                                if token:
                                    yield token
        except httpx.HTTPError as exc:
            logger.error("vllm_stream_error", error=str(exc))
            raise LLMError(f"vLLM streaming failed: {exc}") from exc

    # -- chat (messages list) ------------------------------------------------

    async def chat(self, messages: list[dict], **kwargs) -> str:
        """vLLM /v1/chat/completions – OpenAI-compatible."""
        url = f"{self._base_url}/v1/chat/completions"
        payload = {
            "model": self._model,
            "messages": messages,
            "max_tokens": kwargs.get("max_tokens", self._max_tokens),
            "temperature": kwargs.get("temperature", 0.7),
            "stream": False,
        }
        try:
            async with httpx.AsyncClient(timeout=120) as client:
                response = await client.post(url, json=payload)
                response.raise_for_status()
                choices = response.json().get("choices", [])
                if choices:
                    return choices[0].get("message", {}).get("content", "")
                return ""
        except httpx.HTTPError as exc:
            logger.error("vllm_chat_error", error=str(exc))
            raise LLMError(f"vLLM chat failed: {exc}") from exc

    async def chat_stream(self, messages: list[dict], **kwargs) -> AsyncGenerator[str, None]:
        """vLLM /v1/chat/completions streaming – yields content deltas."""
        url = f"{self._base_url}/v1/chat/completions"
        payload = {
            "model": self._model,
            "messages": messages,
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
                            data = _json.loads(chunk)
                            choices = data.get("choices", [])
                            if choices:
                                delta = choices[0].get("delta", {})
                                token = delta.get("content", "")
                                if token:
                                    yield token
        except httpx.HTTPError as exc:
            logger.error("vllm_chat_stream_error", error=str(exc))
            raise LLMError(f"vLLM chat streaming failed: {exc}") from exc

    # -- health --------------------------------------------------------------

    async def health_check(self) -> bool:
        try:
            async with httpx.AsyncClient(timeout=5) as client:
                response = await client.get(f"{self._base_url}/v1/models")
                return response.status_code == 200
        except httpx.HTTPError:
            return False
