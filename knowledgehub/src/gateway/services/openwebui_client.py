"""
Async HTTP client for the Open WebUI REST API.

Provides typed methods for the subset of Open WebUI endpoints that
KnowledgeHub needs: chat retrieval, user lookup, tag management, and
chat listing.

The client is **resilient by design**: every call catches network and HTTP
errors, logs a warning, and returns ``None`` so callers can degrade
gracefully when Open WebUI is unreachable.

Usage::

    from src.gateway.services.openwebui_client import get_openwebui_client

    client = get_openwebui_client()
    chat = await client.get_chat("abc-123")
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Any

import httpx

from src.config import get_settings
from src.config.logging import get_logger

logger = get_logger(__name__)

# Default timeout for individual requests (seconds).
_DEFAULT_TIMEOUT = 10.0

# Maximum number of automatic retries on transient errors.
_MAX_RETRIES = 2


# ═══════════════════════════════════════════════════════════════════════════════
# Data containers
# ═══════════════════════════════════════════════════════════════════════════════


@dataclass(frozen=True, slots=True)
class OpenWebUIChat:
    """Minimal representation of an Open WebUI chat object."""

    id: str
    title: str
    user_id: str
    created_at: float | None = None
    updated_at: float | None = None
    tags: list[str] = field(default_factory=list)


@dataclass(frozen=True, slots=True)
class OpenWebUIUser:
    """Minimal representation of an Open WebUI user object."""

    id: str
    name: str
    email: str = ""
    role: str = "user"
    created_at: float | None = None


# ═══════════════════════════════════════════════════════════════════════════════
# Rate-limiter (simple token-bucket)
# ═══════════════════════════════════════════════════════════════════════════════


class _RateLimiter:
    """Simple token-bucket rate limiter (per-instance, not distributed)."""

    def __init__(self, max_tokens: int = 30, refill_seconds: float = 60.0) -> None:
        self._max = max_tokens
        self._tokens = float(max_tokens)
        self._refill_rate = max_tokens / refill_seconds
        self._last_refill = time.monotonic()

    def acquire(self) -> bool:
        """Try to consume one token.  Returns *False* if rate-limited."""
        now = time.monotonic()
        elapsed = now - self._last_refill
        self._tokens = min(self._max, self._tokens + elapsed * self._refill_rate)
        self._last_refill = now

        if self._tokens >= 1.0:
            self._tokens -= 1.0
            return True
        return False


# ═══════════════════════════════════════════════════════════════════════════════
# Client
# ═══════════════════════════════════════════════════════════════════════════════


class OpenWebUIClient:
    """Async HTTP client for the Open WebUI API.

    Parameters
    ----------
    base_url:
        Open WebUI base URL (e.g. ``http://open-webui:8080``).
    api_token:
        Bearer token for Open WebUI API authentication.
    timeout:
        Per-request HTTP timeout in seconds.
    """

    def __init__(
        self,
        base_url: str = "http://localhost:3000",
        api_token: str = "",
        timeout: float = _DEFAULT_TIMEOUT,
    ) -> None:
        self._base_url = base_url.rstrip("/")
        self._api_token = api_token
        self._timeout = timeout
        self._limiter = _RateLimiter()
        self._client: httpx.AsyncClient | None = None
        self._available: bool = True  # optimistic

    # ── HTTP transport ─────────────────────────────────────────────────────

    async def _get_client(self) -> httpx.AsyncClient:
        """Lazy-init the underlying httpx client."""
        if self._client is None or self._client.is_closed:
            headers: dict[str, str] = {"Accept": "application/json"}
            if self._api_token:
                headers["Authorization"] = f"Bearer {self._api_token}"
            self._client = httpx.AsyncClient(
                base_url=self._base_url,
                headers=headers,
                timeout=self._timeout,
            )
        return self._client

    async def close(self) -> None:
        """Shutdown the underlying HTTP client."""
        if self._client is not None and not self._client.is_closed:
            await self._client.aclose()
            self._client = None

    async def _request(
        self,
        method: str,
        path: str,
        *,
        json: Any | None = None,
    ) -> dict | list | None:
        """Execute an HTTP request with rate-limiting, retries, and error handling.

        Returns the parsed JSON body on success, or ``None`` on failure.
        """
        if not self._limiter.acquire():
            logger.warning("openwebui_rate_limited", path=path)
            return None

        client = await self._get_client()

        last_exc: Exception | None = None
        for attempt in range(_MAX_RETRIES + 1):
            try:
                resp = await client.request(method, path, json=json)

                if resp.status_code == 404:
                    logger.debug("openwebui_not_found", path=path)
                    self._available = True
                    return None

                resp.raise_for_status()
                self._available = True
                return resp.json()

            except httpx.TimeoutException as exc:
                last_exc = exc
                logger.warning(
                    "openwebui_timeout",
                    path=path,
                    attempt=attempt + 1,
                )
            except httpx.ConnectError as exc:
                last_exc = exc
                self._available = False
                logger.warning(
                    "openwebui_connect_error",
                    path=path,
                    error=str(exc),
                    attempt=attempt + 1,
                )
                break  # don't retry connection errors – host is down
            except httpx.HTTPStatusError as exc:
                last_exc = exc
                logger.warning(
                    "openwebui_http_error",
                    path=path,
                    status=exc.response.status_code,
                    attempt=attempt + 1,
                )
                if exc.response.status_code < 500:
                    break  # don't retry client errors
            except Exception as exc:  # noqa: BLE001
                last_exc = exc
                logger.warning(
                    "openwebui_unexpected_error",
                    path=path,
                    error=str(exc),
                    attempt=attempt + 1,
                )
                break

        logger.warning(
            "openwebui_request_failed",
            path=path,
            error=str(last_exc),
        )
        return None

    # ── Public API ─────────────────────────────────────────────────────────

    @property
    def is_available(self) -> bool:
        """Whether the last request succeeded (optimistic flag)."""
        return self._available

    async def health_check(self) -> bool:
        """Return *True* if Open WebUI is reachable."""
        data = await self._request("GET", "/health")
        ok = data is not None
        self._available = ok
        return ok

    async def get_chat(self, chat_id: str) -> OpenWebUIChat | None:
        """Retrieve a single chat by ID.

        Open WebUI API: ``GET /api/v1/chats/{chat_id}``
        """
        data = await self._request("GET", f"/api/v1/chats/{chat_id}")
        if data is None:
            return None
        return self._parse_chat(data)

    async def get_user(self, user_id: str) -> OpenWebUIUser | None:
        """Retrieve user info by ID.

        Open WebUI API: ``GET /api/v1/users/{user_id}``
        """
        data = await self._request("GET", f"/api/v1/users/{user_id}")
        if data is None:
            return None
        return self._parse_user(data)

    async def update_chat_tags(
        self,
        chat_id: str,
        tags: list[str],
    ) -> bool:
        """Add or replace tags on a chat.

        Open WebUI API: ``POST /api/v1/chats/{chat_id}/tags``

        Returns *True* on success.
        """
        result = await self._request(
            "POST",
            f"/api/v1/chats/{chat_id}/tags",
            json={"tags": tags},
        )
        return result is not None

    async def list_user_chats(
        self,
        user_id: str,
        limit: int = 20,
        skip: int = 0,
    ) -> list[OpenWebUIChat]:
        """List recent chats for a user.

        Open WebUI API: ``GET /api/v1/chats/?user_id=...``
        """
        data = await self._request(
            "GET",
            f"/api/v1/chats/list/user/{user_id}?limit={limit}&skip={skip}",
        )
        if data is None:
            return []
        if isinstance(data, list):
            return [self._parse_chat(c) for c in data if isinstance(c, dict)]
        return []

    # ── Parsers ────────────────────────────────────────────────────────────

    @staticmethod
    def _parse_chat(data: dict) -> OpenWebUIChat:
        """Convert raw JSON dict to :class:`OpenWebUIChat`."""
        # Open WebUI nests the chat object under a "chat" key sometimes
        chat = data.get("chat", data) if isinstance(data.get("chat"), dict) else data
        tags_raw = chat.get("tags") or []
        tags = [t if isinstance(t, str) else t.get("name", "") for t in tags_raw]
        return OpenWebUIChat(
            id=str(chat.get("id", "")),
            title=chat.get("title", ""),
            user_id=str(chat.get("user_id", "")),
            created_at=chat.get("created_at"),
            updated_at=chat.get("updated_at"),
            tags=tags,
        )

    @staticmethod
    def _parse_user(data: dict) -> OpenWebUIUser:
        """Convert raw JSON dict to :class:`OpenWebUIUser`."""
        return OpenWebUIUser(
            id=str(data.get("id", "")),
            name=data.get("name", ""),
            email=data.get("email", ""),
            role=data.get("role", "user"),
            created_at=data.get("created_at"),
        )


# ═══════════════════════════════════════════════════════════════════════════════
# Singleton accessor
# ═══════════════════════════════════════════════════════════════════════════════

_instance: OpenWebUIClient | None = None


def get_openwebui_client() -> OpenWebUIClient:
    """Return a module-level singleton :class:`OpenWebUIClient`.

    Reads ``KNOWLEDGEHUB_OPENWEBUI_URL`` and ``KNOWLEDGEHUB_OPENWEBUI_TOKEN``
    from the environment (via settings) on first call.
    """
    global _instance  # noqa: PLW0603
    if _instance is None:
        settings = get_settings()
        base_url = getattr(settings, "openwebui_url", None) or "http://localhost:3000"
        token = getattr(settings, "openwebui_token", None) or ""
        _instance = OpenWebUIClient(base_url=base_url, api_token=token)
    return _instance
