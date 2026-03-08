"""Tests for the SessionSync service."""

from unittest.mock import AsyncMock, patch

import pytest

from src.gateway.services.openwebui_client import (
    OpenWebUIChat,
    OpenWebUIClient,
    OpenWebUIUser,
)
from src.gateway.services.session_sync import (
    SessionInfo,
    SessionSync,
    UserContext,
    _TTLCache,
)


# ═══════════════════════════════════════════════════════════════════════════════
# TTL Cache
# ═══════════════════════════════════════════════════════════════════════════════


class TestTTLCache:
    def test_put_and_get(self) -> None:
        cache = _TTLCache(maxsize=10, ttl=60.0)
        cache.put("k1", "v1")
        assert cache.get("k1") == "v1"

    def test_get_miss(self) -> None:
        cache = _TTLCache()
        assert cache.get("nonexistent") is None

    def test_ttl_expiry(self) -> None:
        cache = _TTLCache(ttl=0.0)  # immediate expiry
        cache.put("k1", "v1")
        assert cache.get("k1") is None

    def test_maxsize_eviction(self) -> None:
        cache = _TTLCache(maxsize=2, ttl=300.0)
        cache.put("k1", "v1")
        cache.put("k2", "v2")
        cache.put("k3", "v3")  # evicts k1
        assert cache.get("k1") is None
        assert cache.get("k2") == "v2"
        assert cache.get("k3") == "v3"

    def test_invalidate(self) -> None:
        cache = _TTLCache()
        cache.put("k1", "v1")
        cache.invalidate("k1")
        assert cache.get("k1") is None

    def test_clear(self) -> None:
        cache = _TTLCache()
        cache.put("k1", "v1")
        cache.put("k2", "v2")
        cache.clear()
        assert cache.get("k1") is None
        assert cache.get("k2") is None


# ═══════════════════════════════════════════════════════════════════════════════
# SessionSync fixtures
# ═══════════════════════════════════════════════════════════════════════════════


@pytest.fixture
def mock_client() -> OpenWebUIClient:
    client = OpenWebUIClient(base_url="http://fake:3000")
    client._available = True
    return client


@pytest.fixture
def sync(mock_client: OpenWebUIClient) -> SessionSync:
    return SessionSync(client=mock_client, session_ttl=300.0, user_ttl=900.0)


# ═══════════════════════════════════════════════════════════════════════════════
# sync_from_openwebui
# ═══════════════════════════════════════════════════════════════════════════════


@pytest.mark.asyncio
async def test_sync_from_openwebui_success(
    sync: SessionSync, mock_client: OpenWebUIClient
) -> None:
    chat = OpenWebUIChat(
        id="chat-123",
        title="Test Chat",
        user_id="user-abc",
        created_at=1700000000.0,
    )
    with patch.object(mock_client, "get_chat", new_callable=AsyncMock) as mock:
        mock.return_value = chat
        info = await sync.sync_from_openwebui("chat-123")

    assert isinstance(info, SessionInfo)
    assert info.session_id == "chat-123"
    assert info.user_id == "user-abc"
    assert info.chat_title == "Test Chat"
    assert info.synced is True


@pytest.mark.asyncio
async def test_sync_from_openwebui_cached(
    sync: SessionSync, mock_client: OpenWebUIClient
) -> None:
    """Second call should hit cache, not the API."""
    chat = OpenWebUIChat(id="chat-1", title="T", user_id="u1")
    with patch.object(mock_client, "get_chat", new_callable=AsyncMock) as mock:
        mock.return_value = chat
        info1 = await sync.sync_from_openwebui("chat-1")
        info2 = await sync.sync_from_openwebui("chat-1")

    assert mock.call_count == 1  # only one API call
    assert info1.user_id == info2.user_id


@pytest.mark.asyncio
async def test_sync_from_openwebui_degraded(
    sync: SessionSync, mock_client: OpenWebUIClient
) -> None:
    """When Open WebUI is down, return degraded SessionInfo."""
    mock_client._available = False
    with patch.object(mock_client, "get_chat", new_callable=AsyncMock) as mock:
        mock.return_value = None
        info = await sync.sync_from_openwebui("session-xyz")

    assert info.session_id == "session-xyz"
    assert info.user_id == "session-xyz"  # fallback to session_id
    assert info.synced is False


@pytest.mark.asyncio
async def test_sync_from_openwebui_chat_not_found(
    sync: SessionSync, mock_client: OpenWebUIClient
) -> None:
    """Chat not found but Open WebUI is available."""
    mock_client._available = True
    with patch.object(mock_client, "get_chat", new_callable=AsyncMock) as mock:
        mock.return_value = None
        info = await sync.sync_from_openwebui("unknown-chat")

    assert info.synced is False
    assert info.user_id == "unknown-chat"


# ═══════════════════════════════════════════════════════════════════════════════
# get_user_context
# ═══════════════════════════════════════════════════════════════════════════════


@pytest.mark.asyncio
async def test_get_user_context_success(
    sync: SessionSync, mock_client: OpenWebUIClient
) -> None:
    user = OpenWebUIUser(id="u1", name="Alice", email="a@b.com", role="admin")
    chats = [
        OpenWebUIChat(id="c1", title="Chat A", user_id="u1"),
        OpenWebUIChat(id="c2", title="Chat B", user_id="u1"),
    ]
    with (
        patch.object(mock_client, "get_user", new_callable=AsyncMock) as mock_user,
        patch.object(mock_client, "list_user_chats", new_callable=AsyncMock) as mock_chats,
    ):
        mock_user.return_value = user
        mock_chats.return_value = chats
        ctx = await sync.get_user_context("u1")

    assert isinstance(ctx, UserContext)
    assert ctx.user_id == "u1"
    assert ctx.user_name == "Alice"
    assert ctx.email == "a@b.com"
    assert ctx.recent_chat_titles == ["Chat A", "Chat B"]
    assert ctx.total_chats == 2
    assert ctx.synced is True


@pytest.mark.asyncio
async def test_get_user_context_cached(
    sync: SessionSync, mock_client: OpenWebUIClient
) -> None:
    user = OpenWebUIUser(id="u1", name="Bob")
    with (
        patch.object(mock_client, "get_user", new_callable=AsyncMock) as mock_user,
        patch.object(mock_client, "list_user_chats", new_callable=AsyncMock) as mock_chats,
    ):
        mock_user.return_value = user
        mock_chats.return_value = []
        ctx1 = await sync.get_user_context("u1")
        ctx2 = await sync.get_user_context("u1")

    assert mock_user.call_count == 1
    assert ctx1.user_name == ctx2.user_name


@pytest.mark.asyncio
async def test_get_user_context_degraded(
    sync: SessionSync, mock_client: OpenWebUIClient
) -> None:
    mock_client._available = False
    with patch.object(mock_client, "get_user", new_callable=AsyncMock) as mock:
        mock.return_value = None
        ctx = await sync.get_user_context("u1")

    assert ctx.synced is False
    assert ctx.user_id == "u1"
    assert ctx.recent_chat_titles == []


# ═══════════════════════════════════════════════════════════════════════════════
# push_metadata_to_openwebui
# ═══════════════════════════════════════════════════════════════════════════════


@pytest.mark.asyncio
async def test_push_metadata_success(
    sync: SessionSync, mock_client: OpenWebUIClient
) -> None:
    existing_chat = OpenWebUIChat(
        id="chat-1", title="T", user_id="u1", tags=["existing-tag"]
    )
    with (
        patch.object(mock_client, "get_chat", new_callable=AsyncMock) as mock_get,
        patch.object(mock_client, "update_chat_tags", new_callable=AsyncMock) as mock_update,
    ):
        mock_get.return_value = existing_chat
        mock_update.return_value = True
        result = await sync.push_metadata_to_openwebui(
            "chat-1",
            {"contexts": ["IT", "HR"]},
        )

    assert result is True
    # Should merge existing tags with new kh: prefixed tags
    call_args = mock_update.call_args
    tags = set(call_args.args[1])
    assert "existing-tag" in tags
    assert "kh:IT" in tags
    assert "kh:HR" in tags


@pytest.mark.asyncio
async def test_push_metadata_no_contexts(
    sync: SessionSync, mock_client: OpenWebUIClient
) -> None:
    """Empty contexts should be a no-op."""
    result = await sync.push_metadata_to_openwebui("chat-1", {"contexts": []})
    assert result is True  # nothing to push, success


@pytest.mark.asyncio
async def test_push_metadata_chat_not_found(
    sync: SessionSync, mock_client: OpenWebUIClient
) -> None:
    with patch.object(mock_client, "get_chat", new_callable=AsyncMock) as mock:
        mock.return_value = None
        result = await sync.push_metadata_to_openwebui(
            "nonexistent", {"contexts": ["IT"]}
        )

    assert result is False


@pytest.mark.asyncio
async def test_push_metadata_update_fails(
    sync: SessionSync, mock_client: OpenWebUIClient
) -> None:
    chat = OpenWebUIChat(id="c1", title="T", user_id="u1", tags=[])
    with (
        patch.object(mock_client, "get_chat", new_callable=AsyncMock) as mock_get,
        patch.object(mock_client, "update_chat_tags", new_callable=AsyncMock) as mock_update,
    ):
        mock_get.return_value = chat
        mock_update.return_value = False
        result = await sync.push_metadata_to_openwebui(
            "c1", {"detected_contexts": ["Finance"]}
        )

    assert result is False


@pytest.mark.asyncio
async def test_push_metadata_invalidates_cache(
    sync: SessionSync, mock_client: OpenWebUIClient
) -> None:
    """After pushing metadata, session cache should be invalidated."""
    # Pre-populate cache
    sync._session_cache.put("chat-1", SessionInfo(session_id="chat-1", user_id="u1"))

    chat = OpenWebUIChat(id="chat-1", title="T", user_id="u1", tags=[])
    with (
        patch.object(mock_client, "get_chat", new_callable=AsyncMock) as mock_get,
        patch.object(mock_client, "update_chat_tags", new_callable=AsyncMock) as mock_update,
    ):
        mock_get.return_value = chat
        mock_update.return_value = True
        await sync.push_metadata_to_openwebui("chat-1", {"contexts": ["IT"]})

    # Cache should be invalidated
    assert sync._session_cache.get("chat-1") is None


# ═══════════════════════════════════════════════════════════════════════════════
# Cache management
# ═══════════════════════════════════════════════════════════════════════════════


@pytest.mark.asyncio
async def test_clear_caches(
    sync: SessionSync, mock_client: OpenWebUIClient
) -> None:
    sync._session_cache.put("s1", "data")
    sync._user_cache.put("u1", "data")

    sync.clear_caches()

    assert sync._session_cache.get("s1") is None
    assert sync._user_cache.get("u1") is None
