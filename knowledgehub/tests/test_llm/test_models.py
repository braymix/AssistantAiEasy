"""Tests for LLM Pydantic models."""

from src.llm.models import (
    ChatCompletion,
    ChatCompletionChunk,
    ChatMessage,
    Choice,
    ChoiceMessage,
    DeltaMessage,
    HealthStatus,
    ModelDetails,
    ModelInfo,
    RequestMetrics,
    StreamChoice,
    UsageInfo,
)


class TestChatMessage:
    def test_defaults(self) -> None:
        msg = ChatMessage(content="hello")
        assert msg.role == "user"
        assert msg.content == "hello"

    def test_all_roles(self) -> None:
        for role in ("system", "user", "assistant"):
            msg = ChatMessage(role=role, content="x")
            assert msg.role == role


class TestChatCompletion:
    def test_default_fields(self) -> None:
        comp = ChatCompletion(
            model="test",
            choices=[Choice(message=ChoiceMessage(content="hi"))],
        )
        assert comp.object == "chat.completion"
        assert comp.id.startswith("chatcmpl-")
        assert comp.model == "test"
        assert comp.choices[0].message.content == "hi"

    def test_usage_info(self) -> None:
        u = UsageInfo(prompt_tokens=10, completion_tokens=20, total_tokens=30)
        assert u.total_tokens == 30


class TestChatCompletionChunk:
    def test_chunk_structure(self) -> None:
        chunk = ChatCompletionChunk(
            id="test",
            model="m",
            choices=[StreamChoice(delta=DeltaMessage(content="tok"))],
        )
        assert chunk.object == "chat.completion.chunk"
        assert chunk.choices[0].delta.content == "tok"

    def test_stop_chunk(self) -> None:
        chunk = ChatCompletionChunk(
            id="test",
            model="m",
            choices=[StreamChoice(delta=DeltaMessage(), finish_reason="stop")],
        )
        assert chunk.choices[0].finish_reason == "stop"


class TestModelInfo:
    def test_defaults(self) -> None:
        info = ModelInfo(id="llama3:8b")
        assert info.object == "model"
        assert info.owned_by == "knowledgehub"


class TestModelDetails:
    def test_fields(self) -> None:
        d = ModelDetails(id="m", family="llama", parameter_size="8B", quantization="Q4_0")
        assert d.family == "llama"
        assert d.quantization == "Q4_0"


class TestHealthStatus:
    def test_healthy(self) -> None:
        s = HealthStatus(healthy=True, backend="ollama", model="m")
        assert s.healthy is True

    def test_unhealthy_with_detail(self) -> None:
        s = HealthStatus(healthy=False, backend="vllm", detail="connection refused")
        assert s.healthy is False
        assert "connection" in s.detail


class TestRequestMetrics:
    def test_fields(self) -> None:
        m = RequestMetrics(latency_ms=120.5, tokens_generated=50, tokens_per_second=25.3)
        assert m.latency_ms == 120.5
        assert m.tokens_per_second == 25.3
