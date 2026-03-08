"""Tests for the Open WebUI API client."""

import json
import time
from unittest.mock import AsyncMock, patch

import httpx
import pytest

from src.gateway.services.openwebui_client import (
    OpenWebUIChat,
    OpenWebUIClient,
    OpenWebUIUser,
)


# ── Fixtures ───────────────────────────────────────────────────────────────


@pytest.fixture
def client() -> OpenWebUIClient:
    return OpenWebUIClient(
        base_url="http://fake-openwebui:3000",
        api_token="test-token",
        timeout=5.0,
    )


def _mock_response(
    status_code: int = 200,
    json_data: dict | list | None = None,
    text: str = "",
) -> httpx.Response:
    """Build a fake httpx.Response."""
    content = json.dumps(json_data).encode() if json_data is not None else text.encode()
    return httpx.Response(
        status_code=status_code,
        content=content,
        headers={"content-type": "application/json"},
        request=httpx.Request("GET", "http://fake"),
    )


# ── Health check ───────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_health_check_success(client: OpenWebUIClient) -> None:
    with patch.object(client, "_request", new_callable=AsyncMock) as mock:
        mock.return_value = {"status": True}
        result = await client.health_check()
        assert result is True
        assert client.is_available is True


@pytest.mark.asyncio
async def test_health_check_failure(client: OpenWebUIClient) -> None:
    with patch.object(client, "_request", new_callable=AsyncMock) as mock:
        mock.return_value = None
        result = await client.health_check()
        assert result is False
        assert client.is_available is False


# ── get_chat ───────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_get_chat_success(client: OpenWebUIClient) -> None:
    chat_data = {
        "id": "chat-123",
        "title": "My Chat",
        "user_id": "user-456",
        "created_at": 1700000000.0,
        "updated_at": 1700001000.0,
        "tags": ["tag1", "tag2"],
    }
    with patch.object(client, "_request", new_callable=AsyncMock) as mock:
        mock.return_value = chat_data
        chat = await client.get_chat("chat-123")

    assert chat is not None
    assert isinstance(chat, OpenWebUIChat)
    assert chat.id == "chat-123"
    assert chat.title == "My Chat"
    assert chat.user_id == "user-456"
    assert chat.tags == ["tag1", "tag2"]


@pytest.mark.asyncio
async def test_get_chat_nested_format(client: OpenWebUIClient) -> None:
    """Open WebUI sometimes nests chat data under a 'chat' key."""
    nested_data = {
        "chat": {
            "id": "chat-789",
            "title": "Nested Chat",
            "user_id": "user-abc",
            "tags": [],
        }
    }
    with patch.object(client, "_request", new_callable=AsyncMock) as mock:
        mock.return_value = nested_data
        chat = await client.get_chat("chat-789")

    assert chat is not None
    assert chat.id == "chat-789"
    assert chat.title == "Nested Chat"


@pytest.mark.asyncio
async def test_get_chat_not_found(client: OpenWebUIClient) -> None:
    with patch.object(client, "_request", new_callable=AsyncMock) as mock:
        mock.return_value = None
        chat = await client.get_chat("nonexistent")

    assert chat is None


# ── get_user ───────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_get_user_success(client: OpenWebUIClient) -> None:
    user_data = {
        "id": "user-123",
        "name": "Alice",
        "email": "alice@example.com",
        "role": "admin",
        "created_at": 1700000000.0,
    }
    with patch.object(client, "_request", new_callable=AsyncMock) as mock:
        mock.return_value = user_data
        user = await client.get_user("user-123")

    assert user is not None
    assert isinstance(user, OpenWebUIUser)
    assert user.id == "user-123"
    assert user.name == "Alice"
    assert user.email == "alice@example.com"
    assert user.role == "admin"


@pytest.mark.asyncio
async def test_get_user_not_found(client: OpenWebUIClient) -> None:
    with patch.object(client, "_request", new_callable=AsyncMock) as mock:
        mock.return_value = None
        user = await client.get_user("nonexistent")

    assert user is None


# ── update_chat_tags ───────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_update_chat_tags_success(client: OpenWebUIClient) -> None:
    with patch.object(client, "_request", new_callable=AsyncMock) as mock:
        mock.return_value = {"status": "ok"}
        result = await client.update_chat_tags("chat-123", ["kh:IT", "kh:HR"])

    assert result is True
    mock.assert_called_once_with(
        "POST",
        "/api/v1/chats/chat-123/tags",
        json={"tags": ["kh:IT", "kh:HR"]},
    )


@pytest.mark.asyncio
async def test_update_chat_tags_failure(client: OpenWebUIClient) -> None:
    with patch.object(client, "_request", new_callable=AsyncMock) as mock:
        mock.return_value = None
        result = await client.update_chat_tags("chat-123", ["tag"])

    assert result is False


# ── list_user_chats ────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_list_user_chats_success(client: OpenWebUIClient) -> None:
    chats_data = [
        {"id": "c1", "title": "Chat 1", "user_id": "u1", "tags": []},
        {"id": "c2", "title": "Chat 2", "user_id": "u1", "tags": ["t1"]},
    ]
    with patch.object(client, "_request", new_callable=AsyncMock) as mock:
        mock.return_value = chats_data
        chats = await client.list_user_chats("u1", limit=10)

    assert len(chats) == 2
    assert chats[0].id == "c1"
    assert chats[1].title == "Chat 2"


@pytest.mark.asyncio
async def test_list_user_chats_empty(client: OpenWebUIClient) -> None:
    with patch.object(client, "_request", new_callable=AsyncMock) as mock:
        mock.return_value = None
        chats = await client.list_user_chats("u1")

    assert chats == []


# ── Tags parsing ───────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_parse_chat_with_dict_tags(client: OpenWebUIClient) -> None:
    """Open WebUI may return tags as dicts with a 'name' key."""
    data = {
        "id": "c1",
        "title": "T",
        "user_id": "u1",
        "tags": [{"name": "foo"}, {"name": "bar"}],
    }
    with patch.object(client, "_request", new_callable=AsyncMock) as mock:
        mock.return_value = data
        chat = await client.get_chat("c1")

    assert chat is not None
    assert chat.tags == ["foo", "bar"]


# ── Rate limiting ──────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_rate_limiter_blocks(client: OpenWebUIClient) -> None:
    """Exhaust the rate limiter and verify requests are rejected."""
    # Drain all tokens and set last_refill to now so no tokens are refilled
    client._limiter._tokens = 0.0
    client._limiter._last_refill = time.monotonic()

    with patch.object(client, "_get_client", new_callable=AsyncMock) as mock_get:
        result = await client._request("GET", "/test")

    assert result is None
    mock_get.assert_not_called()


# ── Close ──────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_close_idempotent(client: OpenWebUIClient) -> None:
    """Closing a never-opened client should not raise."""
    await client.close()
    await client.close()  # second call is safe
