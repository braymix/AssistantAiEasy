"""Tests for chat schemas and models endpoint."""

import pytest
from httpx import AsyncClient

from src.gateway.schemas.chat import (
    ChatCompletionChunk,
    ChatCompletionRequest,
    ChatCompletionResponse,
    ChatMessage,
    Choice,
    ChoiceMessage,
    DeltaMessage,
    StreamChoice,
)


# ── Schema tests ──────────────────────────────────────────────────────────

def test_chat_completion_request_defaults():
    req = ChatCompletionRequest(
        messages=[ChatMessage(role="user", content="Hello")]
    )
    assert req.model == ""
    assert req.temperature == 0.7
    assert req.stream is False
    assert len(req.messages) == 1
    assert req.messages[0].role == "user"


def test_chat_completion_request_full():
    req = ChatCompletionRequest(
        model="llama3.2:3b",
        messages=[
            ChatMessage(role="system", content="You are helpful"),
            ChatMessage(role="user", content="What is Python?"),
        ],
        temperature=0.5,
        max_tokens=100,
        stream=True,
        user="test-user",
    )
    assert req.model == "llama3.2:3b"
    assert req.stream is True
    assert req.max_tokens == 100
    assert len(req.messages) == 2


def test_chat_completion_response_structure():
    resp = ChatCompletionResponse(
        model="test-model",
        choices=[Choice(message=ChoiceMessage(content="Hello!"))],
    )
    assert resp.object == "chat.completion"
    assert resp.id.startswith("chatcmpl-")
    assert resp.choices[0].message.role == "assistant"
    assert resp.choices[0].message.content == "Hello!"
    assert resp.choices[0].finish_reason == "stop"


def test_streaming_chunk_structure():
    chunk = ChatCompletionChunk(
        id="chatcmpl-test",
        model="test-model",
        choices=[StreamChoice(delta=DeltaMessage(content="Hi"))],
    )
    assert chunk.object == "chat.completion.chunk"
    assert chunk.choices[0].delta.content == "Hi"
    assert chunk.choices[0].finish_reason is None


def test_streaming_stop_chunk():
    chunk = ChatCompletionChunk(
        id="chatcmpl-test",
        model="test-model",
        choices=[StreamChoice(delta=DeltaMessage(), finish_reason="stop")],
    )
    assert chunk.choices[0].finish_reason == "stop"
    assert chunk.choices[0].delta.content is None


# ── Endpoint tests ────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_list_models(client: AsyncClient):
    response = await client.get("/v1/models")
    assert response.status_code == 200
    data = response.json()
    assert data["object"] == "list"
    assert len(data["data"]) >= 1
    assert "id" in data["data"][0]
    assert data["data"][0]["object"] == "model"
