"""
Session synchronisation between KnowledgeHub and Open WebUI.

Maintains a local cache of session/user information fetched from Open WebUI,
and provides helpers for pushing metadata (detected context tags) back.

Designed to degrade gracefully: when Open WebUI is unreachable, the service
uses the raw ``session_id`` as a user identifier, logs a warning, and
continues operating without sync.

Usage::

    from src.gateway.services.session_sync import get_session_sync

    sync = get_session_sync()
    info = await sync.sync_from_openwebui("chat-abc-123")
    ctx  = await sync.get_user_context(info.user_id)
    await sync.push_metadata_to_openwebui("chat-abc-123", {"contexts": ["IT"]})
"""

from __future__ import annotations

import time
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Any

from src.config.logging import get_logger
from src.gateway.services.openwebui_client import (
    OpenWebUIClient,
    get_openwebui_client,
)

logger = get_logger(__name__)


# ═══════════════════════════════════════════════════════════════════════════════
# Data containers
# ═══════════════════════════════════════════════════════════════════════════════


@dataclass(slots=True)
class SessionInfo:
    """Cached session information synced from Open WebUI."""

    session_id: str
    user_id: str
    user_name: str = ""
    chat_title: str = ""
    created_at: float | None = None
    synced: bool = False  # True if data came from Open WebUI


@dataclass(slots=True)
class UserContext:
    """Aggregated context for a given user."""

    user_id: str
    user_name: str = ""
    email: str = ""
    role: str = "user"
    recent_chat_titles: list[str] = field(default_factory=list)
    total_chats: int = 0
    synced: bool = False


# ═══════════════════════════════════════════════════════════════════════════════
# TTL-bounded LRU cache
# ═══════════════════════════════════════════════════════════════════════════════


class _TTLCache:
    """Simple dict-based LRU cache with a per-entry TTL.

    Thread-safety is *not* guaranteed; this is fine because the gateway
    runs requests on a single asyncio event loop.
    """

    def __init__(self, maxsize: int = 512, ttl: float = 300.0) -> None:
        self._maxsize = maxsize
        self._ttl = ttl
        self._store: OrderedDict[str, tuple[float, Any]] = OrderedDict()

    def get(self, key: str) -> Any | None:
        entry = self._store.get(key)
        if entry is None:
            return None
        ts, value = entry
        if (time.monotonic() - ts) > self._ttl:
            self._store.pop(key, None)
            return None
        # Move to end (most recently used)
        self._store.move_to_end(key)
        return value

    def put(self, key: str, value: Any) -> None:
        self._store[key] = (time.monotonic(), value)
        self._store.move_to_end(key)
        while len(self._store) > self._maxsize:
            self._store.popitem(last=False)

    def invalidate(self, key: str) -> None:
        self._store.pop(key, None)

    def clear(self) -> None:
        self._store.clear()


# ═══════════════════════════════════════════════════════════════════════════════
# SessionSync service
# ═══════════════════════════════════════════════════════════════════════════════


class SessionSync:
    """Synchronise session and user data between KnowledgeHub and Open WebUI.

    Parameters
    ----------
    client:
        An :class:`OpenWebUIClient` instance (or *None* to use the singleton).
    session_ttl:
        Cache TTL for session info (seconds).  Default 5 minutes.
    user_ttl:
        Cache TTL for user context (seconds).  Default 15 minutes
        (users change rarely).
    """

    def __init__(
        self,
        client: OpenWebUIClient | None = None,
        session_ttl: float = 300.0,
        user_ttl: float = 900.0,
    ) -> None:
        self._client = client or get_openwebui_client()
        self._session_cache = _TTLCache(maxsize=1024, ttl=session_ttl)
        self._user_cache = _TTLCache(maxsize=256, ttl=user_ttl)

    # ── 1. sync_from_openwebui ─────────────────────────────────────────────

    async def sync_from_openwebui(self, session_id: str) -> SessionInfo:
        """Fetch session/chat info from Open WebUI and return a cached
        :class:`SessionInfo`.

        If Open WebUI is unreachable, returns a *degraded* ``SessionInfo``
        using the raw ``session_id`` as ``user_id``.
        """
        # Check cache first
        cached: SessionInfo | None = self._session_cache.get(session_id)
        if cached is not None:
            logger.debug("session_cache_hit", session_id=session_id)
            return cached

        # Try to fetch from Open WebUI
        chat = await self._client.get_chat(session_id)

        if chat is not None:
            info = SessionInfo(
                session_id=session_id,
                user_id=chat.user_id,
                user_name="",  # populated by get_user_context if needed
                chat_title=chat.title,
                created_at=chat.created_at,
                synced=True,
            )
            logger.info(
                "session_synced",
                session_id=session_id,
                user_id=chat.user_id,
                title=chat.title,
            )
        else:
            # Degraded mode: use session_id as user identifier
            info = SessionInfo(
                session_id=session_id,
                user_id=session_id,
                synced=False,
            )
            if not self._client.is_available:
                logger.warning(
                    "session_sync_degraded",
                    session_id=session_id,
                    reason="Open WebUI unreachable",
                )
            else:
                logger.debug(
                    "session_sync_no_chat",
                    session_id=session_id,
                    reason="chat not found in Open WebUI",
                )

        self._session_cache.put(session_id, info)
        return info

    # ── 2. get_user_context ────────────────────────────────────────────────

    async def get_user_context(self, user_id: str) -> UserContext:
        """Build an aggregated :class:`UserContext` for the given user.

        Fetches user profile and recent chat history from Open WebUI.
        Results are cached for ``user_ttl`` seconds.
        """
        cached: UserContext | None = self._user_cache.get(user_id)
        if cached is not None:
            logger.debug("user_context_cache_hit", user_id=user_id)
            return cached

        # Fetch user info
        user = await self._client.get_user(user_id)

        if user is not None:
            # Fetch recent chats for conversation history summary
            chats = await self._client.list_user_chats(user_id, limit=10)
            recent_titles = [c.title for c in chats if c.title]

            ctx = UserContext(
                user_id=user_id,
                user_name=user.name,
                email=user.email,
                role=user.role,
                recent_chat_titles=recent_titles,
                total_chats=len(chats),
                synced=True,
            )
            logger.info(
                "user_context_synced",
                user_id=user_id,
                name=user.name,
                recent_chats=len(recent_titles),
            )
        else:
            # Degraded mode
            ctx = UserContext(user_id=user_id, synced=False)
            if not self._client.is_available:
                logger.warning(
                    "user_context_degraded",
                    user_id=user_id,
                    reason="Open WebUI unreachable",
                )

        self._user_cache.put(user_id, ctx)
        return ctx

    # ── 3. push_metadata_to_openwebui ──────────────────────────────────────

    async def push_metadata_to_openwebui(
        self,
        session_id: str,
        metadata: dict[str, Any],
    ) -> bool:
        """Push metadata back to Open WebUI as chat tags.

        Extracts context names from *metadata* and calls the Open WebUI
        API to update the chat's tags.  Existing tags are preserved; new
        tags are appended.

        Returns *True* if the update succeeded.
        """
        # Extract context tags from metadata
        contexts: list[str] = metadata.get("contexts", [])
        detected: list[str] = metadata.get("detected_contexts", [])
        tags_to_add = [f"kh:{c}" for c in (*contexts, *detected) if c]

        if not tags_to_add:
            return True  # nothing to push

        # Merge with existing tags
        chat = await self._client.get_chat(session_id)
        if chat is None:
            logger.debug(
                "push_metadata_skip",
                session_id=session_id,
                reason="chat not found",
            )
            return False

        existing_tags = set(chat.tags)
        merged = list(existing_tags | set(tags_to_add))

        ok = await self._client.update_chat_tags(session_id, merged)

        if ok:
            # Invalidate session cache so next fetch picks up new tags
            self._session_cache.invalidate(session_id)
            logger.info(
                "metadata_pushed",
                session_id=session_id,
                new_tags=tags_to_add,
            )
        else:
            logger.warning(
                "metadata_push_failed",
                session_id=session_id,
            )

        return ok

    # ── Cache management ───────────────────────────────────────────────────

    def clear_caches(self) -> None:
        """Clear all in-memory caches."""
        self._session_cache.clear()
        self._user_cache.clear()
        logger.info("session_sync_caches_cleared")


# ═══════════════════════════════════════════════════════════════════════════════
# Singleton accessor
# ═══════════════════════════════════════════════════════════════════════════════

_instance: SessionSync | None = None


def get_session_sync() -> SessionSync:
    """Return a module-level singleton :class:`SessionSync`."""
    global _instance  # noqa: PLW0603
    if _instance is None:
        _instance = SessionSync()
    return _instance
