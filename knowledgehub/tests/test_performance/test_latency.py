"""Performance tests for KnowledgeHub.

Verifies latency constraints and concurrent request handling.
Uses in-memory SQLite and mock services — measures framework overhead only.
"""

from __future__ import annotations

import asyncio
import time
from unittest.mock import AsyncMock

import pytest

from src.detection.engine import DetectionEngine
from src.detection.rules import DetectionContext, KeywordRule, RegexRule
from src.knowledge.service import KnowledgeService
from src.shared.models import DetectionRule, RuleType
from tests.conftest import FakeEmbedder, FakeVectorStore


# ═══════════════════════════════════════════════════════════════════════════════
# 1. Detection latency
# ═══════════════════════════════════════════════════════════════════════════════


@pytest.mark.asyncio
async def test_detection_latency_under_100ms(db_session):
    """Detection with 10 keyword + 5 regex rules must complete under 100ms."""
    # Seed 10 keyword rules
    for i in range(10):
        db_session.add(DetectionRule(
            name=f"kw_rule_{i}",
            rule_type=RuleType.KEYWORD,
            rule_config={"keywords": [f"keyword_{i}", f"term_{i}", f"word_{i}"]},
            target_contexts=[f"ctx_{i}"],
            priority=i,
            enabled=True,
        ))
    # Seed 5 regex rules
    for i in range(5):
        db_session.add(DetectionRule(
            name=f"re_rule_{i}",
            rule_type=RuleType.REGEX,
            rule_config={"pattern": rf"ERR-{i}\d+"},
            target_contexts=[f"err_ctx_{i}"],
            priority=i,
            enabled=True,
        ))
    await db_session.flush()

    engine = DetectionEngine(db_session)

    # Warm up
    await engine.detect("test warm up keyword_0")

    # Measure
    start = time.perf_counter()
    iterations = 20
    for _ in range(iterations):
        await engine.detect("I have a keyword_3 and ERR-42 in my system")
    elapsed_ms = (time.perf_counter() - start) * 1000

    avg_ms = elapsed_ms / iterations
    assert avg_ms < 100, f"Average detection latency {avg_ms:.1f}ms exceeds 100ms"


# ═══════════════════════════════════════════════════════════════════════════════
# 2. RAG query latency
# ═══════════════════════════════════════════════════════════════════════════════


@pytest.mark.asyncio
async def test_rag_query_latency(db_session):
    """RAG build_rag_context with mock vectorstore must complete quickly."""
    vectorstore = FakeVectorStore()
    embedder = FakeEmbedder()
    svc = KnowledgeService(
        session=db_session,
        vectorstore=vectorstore,
        embedder=embedder,
    )

    # Seed 50 knowledge items
    for i in range(50):
        await svc.add_knowledge(
            content=f"Knowledge item number {i} about topic {i % 5}",
            contexts=[f"topic_{i % 5}"],
            source_type="manual",
        )

    # Measure RAG context build
    start = time.perf_counter()
    iterations = 10
    for _ in range(iterations):
        await svc.build_rag_context(
            query="Tell me about topic 3",
            detected_contexts=["topic_3"],
        )
    elapsed_ms = (time.perf_counter() - start) * 1000

    avg_ms = elapsed_ms / iterations
    assert avg_ms < 200, f"Average RAG latency {avg_ms:.1f}ms exceeds 200ms"


# ═══════════════════════════════════════════════════════════════════════════════
# 3. Concurrent requests
# ═══════════════════════════════════════════════════════════════════════════════


@pytest.mark.asyncio
async def test_concurrent_requests(db_session):
    """Multiple concurrent detection requests should not block each other."""
    # Seed rules
    db_session.add(DetectionRule(
        name="concurrent_kw",
        rule_type=RuleType.KEYWORD,
        rule_config={"keywords": ["database", "query"]},
        target_contexts=["database"],
        priority=10,
        enabled=True,
    ))
    await db_session.flush()

    engine = DetectionEngine(db_session)

    async def single_detect(text: str):
        return await engine.detect(text)

    texts = [
        "How to optimize database query?",
        "What is the weather like?",
        "Deploy the database migration",
        "Configure SQL connection pooling",
        "Random unrelated question",
        "Database performance tuning",
        "Error in query execution",
        "Weather forecast for tomorrow",
        "SQL join optimization tips",
        "How to backup database?",
    ]

    start = time.perf_counter()
    results = await asyncio.gather(*[single_detect(t) for t in texts])
    elapsed_ms = (time.perf_counter() - start) * 1000

    # All tasks completed
    assert len(results) == 10

    # Should be fast when run concurrently
    assert elapsed_ms < 500, f"Concurrent detection took {elapsed_ms:.1f}ms for 10 requests"

    # Verify correctness
    db_results = [r for r in results if "database" in r.suggested_topics]
    assert len(db_results) >= 4  # At least the database-related queries


@pytest.mark.asyncio
async def test_detection_with_injected_rules_performance(db_session):
    """Injected Rule objects should evaluate quickly in parallel."""
    rules = [
        KeywordRule(
            name=f"perf_kw_{i}",
            keywords=[f"keyword_{i}", f"term_{i}"],
            contexts=[f"ctx_{i}"],
            priority=i,
        )
        for i in range(20)
    ] + [
        RegexRule(
            name=f"perf_re_{i}",
            patterns=[rf"PATTERN-{i}\w+"],
            contexts=[f"re_ctx_{i}"],
            priority=i,
        )
        for i in range(10)
    ]

    engine = DetectionEngine(db_session, rules=rules)

    start = time.perf_counter()
    iterations = 50
    for _ in range(iterations):
        await engine.detect("keyword_5 and PATTERN-3abc here")
    elapsed_ms = (time.perf_counter() - start) * 1000

    avg_ms = elapsed_ms / iterations
    assert avg_ms < 50, f"Average injected-rule detection {avg_ms:.1f}ms exceeds 50ms"
