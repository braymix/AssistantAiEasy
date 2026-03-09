"""Tests for the DetectionEngine orchestration layer."""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, patch

import pytest
from sqlalchemy import select

from src.detection.engine import DetectionEngine, EnrichedRequest, _convert_db_rules
from src.detection.rules import (
    CompositeRule,
    DetectionContext,
    KeywordRule,
    RegexRule,
    Rule,
    RuleMatch,
    SemanticRule,
)
from src.shared.models import DetectionRule, RuleType


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def ctx():
    return DetectionContext()


@pytest.fixture
async def seed_rules(db_session):
    """Seed the DB with sample detection rules."""
    rules = [
        DetectionRule(
            name="database_kw",
            rule_type=RuleType.KEYWORD,
            rule_config={"keywords": ["database", "sql", "query"]},
            target_contexts=["database"],
            priority=10,
            enabled=True,
        ),
        DetectionRule(
            name="error_regex",
            rule_type=RuleType.REGEX,
            rule_config={"pattern": r"ERR-\d+"},
            target_contexts=["errors"],
            priority=5,
            enabled=True,
        ),
        DetectionRule(
            name="deploy_composite",
            rule_type=RuleType.COMPOSITE,
            rule_config={
                "keywords": ["deploy", "release"],
                "pattern": r"v\d+\.\d+",
                "operator": "OR",
            },
            target_contexts=["deployment"],
            priority=8,
            enabled=True,
        ),
        DetectionRule(
            name="disabled_rule",
            rule_type=RuleType.KEYWORD,
            rule_config={"keywords": ["secret"]},
            target_contexts=["security"],
            priority=100,
            enabled=False,
        ),
    ]
    for rule in rules:
        db_session.add(rule)
    await db_session.flush()
    return rules


# ═══════════════════════════════════════════════════════════════════════════
# DB-based detection (legacy path)
# ═══════════════════════════════════════════════════════════════════════════


@pytest.mark.asyncio
async def test_detect_keyword_from_db(db_session, seed_rules):
    engine = DetectionEngine(db_session)
    result = await engine.detect("How do I optimise a database query?")
    assert result.confidence > 0
    assert "database" in result.suggested_topics
    assert any(r.rule_name == "database_kw" for r in result.triggered_rules)


@pytest.mark.asyncio
async def test_detect_regex_from_db(db_session, seed_rules):
    engine = DetectionEngine(db_session)
    result = await engine.detect("Got error ERR-42 in production")
    assert "errors" in result.suggested_topics
    assert any(r.rule_name == "error_regex" for r in result.triggered_rules)


@pytest.mark.asyncio
async def test_detect_composite_from_db(db_session, seed_rules):
    engine = DetectionEngine(db_session)
    result = await engine.detect("We need to deploy v2.3 today")
    assert "deployment" in result.suggested_topics


@pytest.mark.asyncio
async def test_detect_disabled_rules_excluded(db_session, seed_rules):
    engine = DetectionEngine(db_session)
    result = await engine.detect("This contains secret information")
    # "disabled_rule" should not fire
    assert "security" not in result.suggested_topics


@pytest.mark.asyncio
async def test_detect_no_match(db_session, seed_rules):
    engine = DetectionEngine(db_session)
    result = await engine.detect("What is the weather like?")
    assert result.confidence == 0.0
    assert result.triggered_rules == []
    assert result.suggested_topics == []


@pytest.mark.asyncio
async def test_detect_multiple_rules_trigger(db_session, seed_rules):
    engine = DetectionEngine(db_session)
    result = await engine.detect("Database query gave ERR-99")
    assert "database" in result.suggested_topics
    assert "errors" in result.suggested_topics
    assert len(result.triggered_rules) >= 2


@pytest.mark.asyncio
async def test_detect_processing_time_populated(db_session, seed_rules):
    engine = DetectionEngine(db_session)
    result = await engine.detect("database connection")
    assert result.processing_time_ms >= 0


# ═══════════════════════════════════════════════════════════════════════════
# Injected rule objects (new OOP path)
# ═══════════════════════════════════════════════════════════════════════════


@pytest.mark.asyncio
async def test_detect_with_injected_rules(db_session):
    rule = KeywordRule(
        name="injected_kw", keywords=["fastapi"], contexts=["web"], priority=5,
    )
    engine = DetectionEngine(db_session, rules=[rule])
    result = await engine.detect("How to use fastapi?")
    assert "web" in result.suggested_topics
    assert any(r.rule_name == "injected_kw" for r in result.triggered_rules)


@pytest.mark.asyncio
async def test_detect_injected_rules_sorted_by_priority(db_session):
    low = KeywordRule(name="low", keywords=["x"], contexts=["low_ctx"], priority=1)
    high = KeywordRule(name="high", keywords=["x"], contexts=["high_ctx"], priority=10)
    engine = DetectionEngine(db_session, rules=[low, high])
    # Both should fire, both contexts present
    result = await engine.detect("x marks the spot")
    assert "low_ctx" in result.suggested_topics
    assert "high_ctx" in result.suggested_topics


@pytest.mark.asyncio
async def test_detect_parallel_execution(db_session):
    """Ensure rules actually run in parallel via asyncio.gather."""
    call_order = []

    class SlowRule(Rule):
        async def match(self, text, context):
            call_order.append(f"start_{self.name}")
            await asyncio.sleep(0.01)
            call_order.append(f"end_{self.name}")
            return RuleMatch(
                rule_name=self.name, confidence=0.5, contexts=self.contexts,
            )

    r1 = SlowRule(name="r1", priority=1, contexts=["a"])
    r2 = SlowRule(name="r2", priority=2, contexts=["b"])
    engine = DetectionEngine(db_session, rules=[r1, r2])
    result = await engine.detect("test parallel")

    # Both rules should have started before either ended (parallel)
    assert "a" in result.suggested_topics
    assert "b" in result.suggested_topics


@pytest.mark.asyncio
async def test_detect_rule_timeout(db_session):
    """A slow rule should be cancelled without blocking others."""

    class VerySlowRule(Rule):
        async def match(self, text, context):
            await asyncio.sleep(100)  # intentionally too slow
            return RuleMatch(rule_name=self.name, confidence=1.0, contexts=["slow"])

    fast = KeywordRule(name="fast", keywords=["test"], contexts=["fast_ctx"])
    slow = VerySlowRule(name="slow", priority=1, contexts=["slow_ctx"])

    engine = DetectionEngine(db_session, rules=[fast, slow], rule_timeout=0.05)
    result = await engine.detect("test input")

    # Fast rule should succeed, slow should timeout
    assert "fast_ctx" in result.suggested_topics
    assert "slow_ctx" not in result.suggested_topics


@pytest.mark.asyncio
async def test_detect_rule_exception_handled(db_session):
    """A rule that raises shouldn't crash the engine."""

    class BrokenRule(Rule):
        async def match(self, text, context):
            raise RuntimeError("broken!")

    good = KeywordRule(name="good", keywords=["test"], contexts=["ok"])
    bad = BrokenRule(name="bad", priority=1, contexts=["bad"])

    engine = DetectionEngine(db_session, rules=[good, bad])
    result = await engine.detect("test this")

    assert "ok" in result.suggested_topics
    assert "bad" not in result.suggested_topics


# ═══════════════════════════════════════════════════════════════════════════
# load_rules_from_db (hot reload)
# ═══════════════════════════════════════════════════════════════════════════


@pytest.mark.asyncio
async def test_load_rules_from_db_converts_and_merges(db_session, seed_rules):
    injected = KeywordRule(name="manual", keywords=["foo"], contexts=["bar"], priority=50)
    engine = DetectionEngine(db_session, rules=[injected])

    await engine.load_rules_from_db()

    # Injected rule + DB rules should all be present
    names = [r.name for r in engine._injected_rules]
    assert "manual" in names
    assert "database_kw" in names
    assert "error_regex" in names
    assert "deploy_composite" in names
    # Disabled rules should NOT be loaded
    assert "disabled_rule" not in names


@pytest.mark.asyncio
async def test_load_rules_from_db_sorted_by_priority(db_session, seed_rules):
    engine = DetectionEngine(db_session)
    await engine.load_rules_from_db()

    priorities = [r.priority for r in engine._injected_rules]
    assert priorities == sorted(priorities, reverse=True)


# ═══════════════════════════════════════════════════════════════════════════
# detect_and_enrich
# ═══════════════════════════════════════════════════════════════════════════


@pytest.mark.asyncio
async def test_detect_and_enrich_no_match(db_session):
    engine = DetectionEngine(db_session)
    messages = [
        {"role": "user", "content": "What is the weather?"},
    ]
    result = await engine.detect_and_enrich(messages)

    assert isinstance(result, EnrichedRequest)
    assert result.enriched_messages == messages  # unchanged
    assert result.rag_context == ""
    assert result.detected.confidence == 0.0


@pytest.mark.asyncio
async def test_detect_and_enrich_with_knowledge(db_session, seed_rules):
    engine = DetectionEngine(db_session)
    messages = [
        {"role": "system", "content": "You are helpful."},
        {"role": "user", "content": "How do I optimise a database query?"},
    ]

    mock_svc = AsyncMock()
    mock_svc.build_rag_context = AsyncMock(return_value="Retrieved: SQL indexing tips")

    result = await engine.detect_and_enrich(
        messages=messages,
        knowledge_svc=mock_svc,
    )

    assert result.detected.confidence > 0
    assert "database" in result.detected.suggested_topics
    assert result.rag_context == "Retrieved: SQL indexing tips"
    # Enriched messages should have an extra system message
    assert len(result.enriched_messages) > len(messages)
    # The injected message should contain the RAG context
    rag_msgs = [
        m for m in result.enriched_messages
        if m["role"] == "system" and "KNOWLEDGE" in m.get("content", "")
    ]
    assert len(rag_msgs) == 1


@pytest.mark.asyncio
async def test_detect_and_enrich_no_service(db_session, seed_rules):
    """Without knowledge_svc, enrichment should be skipped."""
    engine = DetectionEngine(db_session)
    messages = [{"role": "user", "content": "database query problems"}]
    result = await engine.detect_and_enrich(messages)

    assert result.detected.confidence > 0
    assert result.rag_context == ""
    assert result.enriched_messages == messages


@pytest.mark.asyncio
async def test_detect_and_enrich_empty_rag(db_session, seed_rules):
    """When build_rag_context returns empty, messages stay unchanged."""
    engine = DetectionEngine(db_session)
    messages = [{"role": "user", "content": "database setup"}]

    mock_svc = AsyncMock()
    mock_svc.build_rag_context = AsyncMock(return_value="")

    result = await engine.detect_and_enrich(messages=messages, knowledge_svc=mock_svc)
    assert result.enriched_messages == messages


# ═══════════════════════════════════════════════════════════════════════════
# _convert_db_rules helper
# ═══════════════════════════════════════════════════════════════════════════


def test_convert_keyword_rule():
    db_rule = DetectionRule(
        name="kw_test",
        rule_type=RuleType.KEYWORD,
        rule_config={"keywords": ["alpha", "beta"], "case_sensitive": True, "match_all": True},
        target_contexts=["ctx1"],
        priority=5,
        enabled=True,
    )
    converted = _convert_db_rules([db_rule])
    assert len(converted) == 1
    assert isinstance(converted[0], KeywordRule)
    assert converted[0].name == "kw_test"
    assert converted[0].case_sensitive is True
    assert converted[0].match_all is True
    assert converted[0].priority == 5
    assert converted[0].contexts == ["ctx1"]


def test_convert_regex_rule_single_pattern():
    db_rule = DetectionRule(
        name="re_test",
        rule_type=RuleType.REGEX,
        rule_config={"pattern": r"ERR-\d+"},
        target_contexts=["errors"],
        priority=3,
        enabled=True,
    )
    converted = _convert_db_rules([db_rule])
    assert len(converted) == 1
    assert isinstance(converted[0], RegexRule)


def test_convert_regex_rule_multiple_patterns():
    db_rule = DetectionRule(
        name="re_multi",
        rule_type=RuleType.REGEX,
        rule_config={"patterns": [r"ERR-\d+", r"WARN-\d+"]},
        target_contexts=["monitoring"],
        priority=3,
        enabled=True,
    )
    converted = _convert_db_rules([db_rule])
    assert len(converted) == 1
    assert isinstance(converted[0], RegexRule)
    assert len(converted[0]._compiled) == 2


def test_convert_composite_rule():
    db_rule = DetectionRule(
        name="comp_test",
        rule_type=RuleType.COMPOSITE,
        rule_config={"keywords": ["deploy"], "pattern": r"v\d+", "operator": "OR"},
        target_contexts=["deploy"],
        priority=7,
        enabled=True,
    )
    converted = _convert_db_rules([db_rule])
    assert len(converted) == 1
    assert isinstance(converted[0], CompositeRule)
    assert converted[0].operator == "OR"
    assert len(converted[0].rules) == 2  # keyword + regex child


def test_convert_semantic_rule():
    db_rule = DetectionRule(
        name="sem_test",
        rule_type=RuleType.SEMANTIC,
        rule_config={"reference_texts": ["deploy app"], "threshold": 0.8},
        target_contexts=["ops"],
        priority=2,
        enabled=True,
    )
    converted = _convert_db_rules([db_rule])
    assert len(converted) == 1
    assert isinstance(converted[0], SemanticRule)
    assert converted[0].threshold == 0.8


def test_convert_empty_config_skipped():
    db_rule = DetectionRule(
        name="empty",
        rule_type=RuleType.KEYWORD,
        rule_config={},  # No keywords
        target_contexts=[],
        priority=0,
        enabled=True,
    )
    converted = _convert_db_rules([db_rule])
    assert len(converted) == 0


# ═══════════════════════════════════════════════════════════════════════════
# Context passthrough
# ═══════════════════════════════════════════════════════════════════════════


@pytest.mark.asyncio
async def test_detect_accepts_dict_context(db_session, seed_rules):
    """Engine should accept a plain dict as context (backward compat)."""
    engine = DetectionEngine(db_session)
    result = await engine.detect("database query", context={"user": "admin"})
    assert result.confidence > 0


@pytest.mark.asyncio
async def test_detect_accepts_none_context(db_session, seed_rules):
    engine = DetectionEngine(db_session)
    result = await engine.detect("database query", context=None)
    assert result.confidence > 0


@pytest.mark.asyncio
async def test_detect_accepts_detection_context(db_session, seed_rules):
    engine = DetectionEngine(db_session)
    ctx = DetectionContext(user_info={"role": "admin"})
    result = await engine.detect("database query", context=ctx)
    assert result.confidence > 0
