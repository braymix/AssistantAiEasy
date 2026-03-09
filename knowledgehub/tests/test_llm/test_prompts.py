"""Tests for prompt templates and token utilities."""

from src.llm.prompts import (
    KNOWLEDGE_EXTRACTION_PROMPT,
    RAG_SYSTEM_PROMPT,
    RERANK_PROMPT,
    SOURCE_ATTRIBUTION_HEADER,
    SUMMARIZE_PROMPT,
    estimate_tokens,
    truncate_to_tokens,
)


class TestEstimateTokens:
    def test_empty_string(self) -> None:
        assert estimate_tokens("") == 1  # min 1

    def test_short_string(self) -> None:
        # "hello" = 5 chars / 4 = 1.25 → 1
        assert estimate_tokens("hello") == 1

    def test_longer_string(self) -> None:
        text = "A" * 400
        assert estimate_tokens(text) == 100


class TestTruncateToTokens:
    def test_short_text_unchanged(self) -> None:
        text = "short text"
        assert truncate_to_tokens(text, 100) == text

    def test_long_text_truncated(self) -> None:
        text = "word " * 200  # 1000 chars
        result = truncate_to_tokens(text, 50)  # ~200 chars budget
        assert len(result) <= 210  # some tolerance for word boundary
        assert result.endswith("…")

    def test_truncation_at_word_boundary(self) -> None:
        text = "hello world this is a test"
        result = truncate_to_tokens(text, 3)  # ~12 chars budget
        assert "…" in result
        # Should not cut in the middle of a word


class TestPromptTemplates:
    def test_rag_system_prompt_has_placeholder(self) -> None:
        assert "{knowledge}" in RAG_SYSTEM_PROMPT

    def test_rag_system_prompt_formats(self) -> None:
        result = RAG_SYSTEM_PROMPT.format(knowledge="Test knowledge")
        assert "Test knowledge" in result
        assert "KNOWLEDGE BASE" in result

    def test_extraction_prompt_valid(self) -> None:
        assert "JSON" in KNOWLEDGE_EXTRACTION_PROMPT
        assert "confidence" in KNOWLEDGE_EXTRACTION_PROMPT

    def test_rerank_prompt_has_placeholders(self) -> None:
        assert "{query}" in RERANK_PROMPT
        assert "{excerpts}" in RERANK_PROMPT

    def test_summarize_prompt_has_placeholder(self) -> None:
        assert "{text}" in SUMMARIZE_PROMPT

    def test_source_attribution_header(self) -> None:
        assert "Fonti" in SOURCE_ATTRIBUTION_HEADER
