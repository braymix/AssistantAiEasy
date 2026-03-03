"""Tests for detection rule evaluation – legacy functions and new Rule classes."""

import pytest

from src.detection.rules import (
    CompositeRule,
    DetectionContext,
    KeywordRule,
    LLMRule,
    RegexRule,
    RuleMatch,
    SemanticRule,
    _cosine_similarity,
    _parse_llm_classification,
    evaluate_keyword_rule,
    evaluate_pattern_rule,
)


# ═══════════════════════════════════════════════════════════════════════════
# Legacy evaluation functions (backward-compatible tests)
# ═══════════════════════════════════════════════════════════════════════════


def test_keyword_rule_match():
    match = evaluate_keyword_rule(
        text="How do I configure the database connection?",
        rule_id="r1",
        rule_name="database_config",
        keywords=["database", "connection", "config"],
    )
    assert match.matched is True
    assert match.confidence > 0
    assert "database" in match.matched_keywords
    assert "connection" in match.matched_keywords


def test_keyword_rule_no_match():
    match = evaluate_keyword_rule(
        text="What is the weather today?",
        rule_id="r1",
        rule_name="database_config",
        keywords=["database", "connection"],
    )
    assert match.matched is False
    assert match.confidence == 0.0


def test_keyword_rule_case_insensitive():
    match = evaluate_keyword_rule(
        text="DATABASE connection pooling",
        rule_id="r1",
        rule_name="db",
        keywords=["database"],
    )
    assert match.matched is True


def test_pattern_rule_match():
    match = evaluate_pattern_rule(
        text="Error code: ERR-12345 occurred",
        rule_id="r2",
        rule_name="error_code",
        pattern=r"ERR-\d+",
    )
    assert match.matched is True
    assert match.confidence == 0.8
    assert "ERR-12345" in match.matched_keywords


def test_pattern_rule_no_match():
    match = evaluate_pattern_rule(
        text="Everything is working fine",
        rule_id="r2",
        rule_name="error_code",
        pattern=r"ERR-\d+",
    )
    assert match.matched is False


def test_pattern_rule_invalid_regex():
    match = evaluate_pattern_rule(
        text="some text",
        rule_id="r3",
        rule_name="broken",
        pattern=r"[invalid",
    )
    assert match.matched is False


# ═══════════════════════════════════════════════════════════════════════════
# 1. KeywordRule (new class)
# ═══════════════════════════════════════════════════════════════════════════


@pytest.mark.asyncio
async def test_keyword_rule_class_or_mode():
    rule = KeywordRule(
        name="test_kw", keywords=["python", "java"], contexts=["tech"],
    )
    ctx = DetectionContext()
    result = await rule.match("I love python programming", ctx)
    assert result is not None
    assert result.rule_name == "test_kw"
    assert result.confidence > 0
    assert "python" in result.extracted["matched_keywords"]
    assert result.contexts == ["tech"]


@pytest.mark.asyncio
async def test_keyword_rule_class_and_mode_all_present():
    rule = KeywordRule(
        name="test_and",
        keywords=["deploy", "docker"],
        match_all=True,
        contexts=["devops"],
    )
    ctx = DetectionContext()
    result = await rule.match("Deploy the app using docker containers", ctx)
    assert result is not None
    assert result.confidence == 1.0


@pytest.mark.asyncio
async def test_keyword_rule_class_and_mode_partial():
    rule = KeywordRule(
        name="test_and",
        keywords=["deploy", "docker", "kubernetes"],
        match_all=True,
    )
    ctx = DetectionContext()
    result = await rule.match("Deploy with docker", ctx)
    # Only 2 of 3 matched — AND mode should fail
    assert result is None


@pytest.mark.asyncio
async def test_keyword_rule_class_no_match():
    rule = KeywordRule(name="test_kw", keywords=["rust", "go"])
    ctx = DetectionContext()
    result = await rule.match("I love python", ctx)
    assert result is None


@pytest.mark.asyncio
async def test_keyword_rule_class_case_sensitive():
    rule = KeywordRule(
        name="case_test",
        keywords=["Python"],
        case_sensitive=True,
    )
    ctx = DetectionContext()
    assert await rule.match("PYTHON is great", ctx) is None
    assert await rule.match("Python is great", ctx) is not None


# ═══════════════════════════════════════════════════════════════════════════
# 2. RegexRule
# ═══════════════════════════════════════════════════════════════════════════


@pytest.mark.asyncio
async def test_regex_rule_match():
    rule = RegexRule(
        name="error_code",
        patterns=[r"ERR-(?P<code>\d+)"],
        contexts=["errors"],
    )
    ctx = DetectionContext()
    result = await rule.match("Got ERR-42 in production", ctx)
    assert result is not None
    assert result.confidence == 0.8
    assert "ERR-42" in result.extracted["matches"]
    assert result.extracted["groups"]["code"] == "42"
    assert result.contexts == ["errors"]


@pytest.mark.asyncio
async def test_regex_rule_no_match():
    rule = RegexRule(name="err", patterns=[r"ERR-\d+"])
    ctx = DetectionContext()
    result = await rule.match("Everything fine", ctx)
    assert result is None


@pytest.mark.asyncio
async def test_regex_rule_multiple_patterns():
    rule = RegexRule(
        name="multi",
        patterns=[r"ERR-\d+", r"WARN-\d+"],
        contexts=["monitoring"],
    )
    ctx = DetectionContext()
    result = await rule.match("Got ERR-1 and WARN-2", ctx)
    assert result is not None
    assert len(result.extracted["matches"]) == 2


@pytest.mark.asyncio
async def test_regex_rule_invalid_pattern_skipped():
    # Invalid pattern "[bad" should be skipped, valid one should still work
    rule = RegexRule(name="mixed", patterns=[r"[bad", r"OK-\d+"])
    ctx = DetectionContext()
    result = await rule.match("Found OK-123", ctx)
    assert result is not None
    assert "OK-123" in result.extracted["matches"]


# ═══════════════════════════════════════════════════════════════════════════
# 3. SemanticRule
# ═══════════════════════════════════════════════════════════════════════════


class FakeEmbedder:
    """Fake embedder that returns deterministic vectors."""

    async def embed(self, texts):
        # Return slightly different vectors for different texts
        return [[float(i + hash(t) % 100) / 100 for i in range(8)] for t in texts]

    async def embed_single(self, text):
        result = await self.embed([text])
        return result[0]

    @property
    def dimension(self):
        return 8


@pytest.mark.asyncio
async def test_semantic_rule_match():
    embedder = FakeEmbedder()
    rule = SemanticRule(
        name="semantic_test",
        reference_texts=["deploy application"],
        threshold=0.0,  # Low threshold so fake embeddings match
        contexts=["devops"],
        embedder=embedder,
    )
    ctx = DetectionContext()
    result = await rule.match("deploy application", ctx)
    assert result is not None
    assert result.rule_name == "semantic_test"
    assert result.contexts == ["devops"]


@pytest.mark.asyncio
async def test_semantic_rule_caches_reference_embeddings():
    embedder = FakeEmbedder()
    rule = SemanticRule(
        name="cache_test",
        reference_texts=["reference one", "reference two"],
        threshold=0.0,
        embedder=embedder,
    )
    ctx = DetectionContext()
    await rule.match("first query", ctx)
    ref1 = rule._reference_embeddings
    await rule.match("second query", ctx)
    ref2 = rule._reference_embeddings
    # Same object — embeddings cached
    assert ref1 is ref2


@pytest.mark.asyncio
async def test_semantic_rule_below_threshold():
    embedder = FakeEmbedder()
    rule = SemanticRule(
        name="strict",
        reference_texts=["deploy"],
        threshold=9999.0,  # Impossibly high
        embedder=embedder,
    )
    ctx = DetectionContext()
    result = await rule.match("something else", ctx)
    assert result is None


# ═══════════════════════════════════════════════════════════════════════════
# 4. CompositeRule
# ═══════════════════════════════════════════════════════════════════════════


@pytest.mark.asyncio
async def test_composite_and_all_match():
    r1 = KeywordRule(name="kw1", keywords=["python"], contexts=["lang"])
    r2 = KeywordRule(name="kw2", keywords=["deploy"], contexts=["ops"])
    composite = CompositeRule(
        name="all_match", rules=[r1, r2], operator="AND", contexts=["full"],
    )
    ctx = DetectionContext()
    result = await composite.match("Deploy the python app", ctx)
    assert result is not None
    assert result.rule_name == "all_match"
    # AND → average confidence
    assert 0 < result.confidence <= 1.0
    # Contexts merged: own ["full"] + children ["lang", "ops"]
    assert "full" in result.contexts
    assert "lang" in result.contexts
    assert "ops" in result.contexts


@pytest.mark.asyncio
async def test_composite_and_partial_fail():
    r1 = KeywordRule(name="kw1", keywords=["python"])
    r2 = KeywordRule(name="kw2", keywords=["golang"])
    composite = CompositeRule(name="partial", rules=[r1, r2], operator="AND")
    ctx = DetectionContext()
    result = await composite.match("I use python", ctx)
    assert result is None  # golang not matched


@pytest.mark.asyncio
async def test_composite_or_at_least_one():
    r1 = KeywordRule(name="kw1", keywords=["python"])
    r2 = KeywordRule(name="kw2", keywords=["golang"])
    composite = CompositeRule(name="any_match", rules=[r1, r2], operator="OR")
    ctx = DetectionContext()
    result = await composite.match("I use python", ctx)
    assert result is not None


@pytest.mark.asyncio
async def test_composite_or_none():
    r1 = KeywordRule(name="kw1", keywords=["rust"])
    r2 = KeywordRule(name="kw2", keywords=["haskell"])
    composite = CompositeRule(name="none", rules=[r1, r2], operator="OR")
    ctx = DetectionContext()
    result = await composite.match("I use python", ctx)
    assert result is None


@pytest.mark.asyncio
async def test_composite_not_no_child_matches():
    r1 = KeywordRule(name="kw1", keywords=["forbidden_word"])
    composite = CompositeRule(
        name="not_rule", rules=[r1], operator="NOT", contexts=["safe"],
    )
    ctx = DetectionContext()
    result = await composite.match("This is a normal sentence", ctx)
    assert result is not None
    assert result.confidence == 1.0
    assert result.contexts == ["safe"]


@pytest.mark.asyncio
async def test_composite_not_child_matches():
    r1 = KeywordRule(name="kw1", keywords=["forbidden"])
    composite = CompositeRule(name="not_rule", rules=[r1], operator="NOT")
    ctx = DetectionContext()
    result = await composite.match("This has forbidden content", ctx)
    assert result is None  # Child matched → NOT fails


def test_composite_invalid_operator():
    with pytest.raises(ValueError, match="Invalid CompositeRule operator"):
        CompositeRule(name="bad", rules=[], operator="XOR")


# ═══════════════════════════════════════════════════════════════════════════
# 5. LLMRule
# ═══════════════════════════════════════════════════════════════════════════


class FakeLLM:
    """Minimal async LLM for testing."""

    def __init__(self, response: str):
        self._response = response
        self.call_count = 0

    async def chat(self, messages, **kwargs):
        self.call_count += 1
        return self._response


@pytest.mark.asyncio
async def test_llm_rule_match():
    llm = FakeLLM('{"match": true, "confidence": 0.9, "reason": "relevant"}')
    rule = LLMRule(
        name="llm_test",
        prompt_template="Does this mention databases? Text: {text}",
        contexts=["database"],
        llm=llm,
    )
    ctx = DetectionContext()
    result = await rule.match("Configure the PostgreSQL pool", ctx)
    assert result is not None
    assert result.confidence == 0.9
    assert result.extracted["reason"] == "relevant"
    assert result.contexts == ["database"]


@pytest.mark.asyncio
async def test_llm_rule_no_match():
    llm = FakeLLM('{"match": false, "confidence": 0.1, "reason": "not relevant"}')
    rule = LLMRule(name="llm_no", prompt_template="Check: {text}", llm=llm)
    ctx = DetectionContext()
    result = await rule.match("Hello world", ctx)
    assert result is None


@pytest.mark.asyncio
async def test_llm_rule_caches_results():
    llm = FakeLLM('{"match": true, "confidence": 0.8, "reason": "ok"}')
    rule = LLMRule(name="llm_cache", prompt_template="{text}", llm=llm)
    ctx = DetectionContext()

    r1 = await rule.match("same input", ctx)
    r2 = await rule.match("same input", ctx)
    assert r1 is not None
    assert r1 is r2  # exact same cached object
    assert llm.call_count == 1  # LLM called only once


@pytest.mark.asyncio
async def test_llm_rule_different_inputs_not_cached():
    llm = FakeLLM('{"match": true, "confidence": 0.8, "reason": "ok"}')
    rule = LLMRule(name="llm_diff", prompt_template="{text}", llm=llm)
    ctx = DetectionContext()

    await rule.match("input a", ctx)
    await rule.match("input b", ctx)
    assert llm.call_count == 2


@pytest.mark.asyncio
async def test_llm_rule_cache_eviction():
    llm = FakeLLM('{"match": true, "confidence": 0.8, "reason": "ok"}')
    rule = LLMRule(
        name="llm_evict", prompt_template="{text}", llm=llm, cache_maxsize=2,
    )
    ctx = DetectionContext()

    await rule.match("a", ctx)
    await rule.match("b", ctx)
    assert len(rule._cache) == 2

    await rule.match("c", ctx)
    assert len(rule._cache) == 2  # oldest evicted


@pytest.mark.asyncio
async def test_llm_rule_handles_exception():
    class BrokenLLM:
        async def chat(self, messages, **kwargs):
            raise RuntimeError("LLM down")

    rule = LLMRule(name="broken_llm", prompt_template="{text}", llm=BrokenLLM())
    ctx = DetectionContext()
    result = await rule.match("test", ctx)
    assert result is None  # graceful failure


# ═══════════════════════════════════════════════════════════════════════════
# Helper functions
# ═══════════════════════════════════════════════════════════════════════════


def test_cosine_similarity_identical():
    v = [1.0, 2.0, 3.0]
    assert abs(_cosine_similarity(v, v) - 1.0) < 1e-6


def test_cosine_similarity_orthogonal():
    a = [1.0, 0.0]
    b = [0.0, 1.0]
    assert abs(_cosine_similarity(a, b)) < 1e-6


def test_cosine_similarity_zero_vector():
    assert _cosine_similarity([0, 0, 0], [1, 2, 3]) == 0.0


def test_parse_llm_classification_direct():
    result = _parse_llm_classification(
        '{"match": true, "confidence": 0.85, "reason": "yes"}',
        "test_rule",
        ["ctx"],
    )
    assert result is not None
    assert result.confidence == 0.85


def test_parse_llm_classification_no_match():
    result = _parse_llm_classification(
        '{"match": false, "confidence": 0.1}',
        "test_rule",
        ["ctx"],
    )
    assert result is None


def test_parse_llm_classification_code_block():
    text = '```json\n{"match": true, "confidence": 0.7}\n```'
    result = _parse_llm_classification(text, "test", [])
    assert result is not None
    assert result.confidence == 0.7


def test_parse_llm_classification_embedded():
    text = 'Based on analysis: {"match": true, "confidence": 0.6} is the result.'
    result = _parse_llm_classification(text, "test", [])
    assert result is not None


def test_parse_llm_classification_invalid():
    result = _parse_llm_classification("not json at all", "test", [])
    assert result is None


# ═══════════════════════════════════════════════════════════════════════════
# DetectionContext
# ═══════════════════════════════════════════════════════════════════════════


def test_detection_context_defaults():
    ctx = DetectionContext()
    assert ctx.conversation_history == []
    assert ctx.user_info == {}


def test_detection_context_with_data():
    ctx = DetectionContext(
        conversation_history=[{"role": "user", "content": "hi"}],
        user_info={"name": "admin"},
    )
    assert len(ctx.conversation_history) == 1
    assert ctx.user_info["name"] == "admin"
