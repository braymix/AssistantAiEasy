"""
Context detection engine – evaluates rules against input text.

Supports two modes:
  1. **DB-based** – loads :class:`DetectionRule` rows from the database
     and converts them to rule objects (original behaviour, preserved for
     the existing chat proxy flow).
  2. **Injected rules** – accepts a pre-built list of :class:`Rule` objects
     (new OOP path with parallel execution, timeouts, and caching).
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from src.config.logging import get_logger
from src.detection.rules import (
    CompositeRule,
    DetectionContext,
    KeywordRule,
    LLMRule,
    RegexRule,
    Rule,
    RuleMatch,
    SemanticRule,
    _LegacyRuleMatch,
    evaluate_keyword_rule,
    evaluate_pattern_rule,
)
from src.gateway.schemas.detection import DetectionResult, TriggeredRule
from src.shared.models import DetectionRule, RuleType

if TYPE_CHECKING:
    from src.gateway.schemas.chat import ChatCompletionRequest
    from src.knowledge.embeddings import EmbeddingProvider
    from src.knowledge.service import KnowledgeService

logger = get_logger(__name__)

# Per-rule timeout (seconds)
_DEFAULT_RULE_TIMEOUT = 5.0

RAG_SYSTEM_TEMPLATE = (
    "Use the following knowledge base excerpts to inform your answer. "
    "If the excerpts are not relevant, ignore them.\n\n"
    "---BEGIN KNOWLEDGE---\n{context}\n---END KNOWLEDGE---"
)


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class EnrichedRequest:
    """Result of :meth:`DetectionEngine.detect_and_enrich`."""

    original_messages: list[dict]
    enriched_messages: list[dict]
    rag_context: str
    detected: DetectionResult


# ═══════════════════════════════════════════════════════════════════════════
# Engine
# ═══════════════════════════════════════════════════════════════════════════


class DetectionEngine:
    """Orchestrates rule evaluation against user input.

    Parameters
    ----------
    session : AsyncSession
        Database session for loading rules from DB.
    rules : list[Rule] | None
        Pre-built rule objects.  When supplied, these are used alongside
        DB-loaded rules.
    rule_timeout : float
        Max seconds an individual rule may take before being cancelled.
    """

    def __init__(
        self,
        session: AsyncSession,
        rules: list[Rule] | None = None,
        rule_timeout: float = _DEFAULT_RULE_TIMEOUT,
    ):
        self._session = session
        self._injected_rules: list[Rule] = sorted(
            rules or [], key=lambda r: r.priority, reverse=True,
        )
        self._rule_timeout = rule_timeout

    # ------------------------------------------------------------------
    # DB rule loading & conversion
    # ------------------------------------------------------------------

    async def _load_db_rules(self) -> list[DetectionRule]:
        """Fetch enabled rules from the database, ordered by priority."""
        q = (
            select(DetectionRule)
            .where(DetectionRule.enabled.is_(True))
            .order_by(DetectionRule.priority.desc())
        )
        result = await self._session.execute(q)
        return list(result.scalars().all())

    async def load_rules_from_db(self) -> None:
        """Hot-reload: load DB rules and convert to Rule objects.

        Merges with any previously injected rules.  Can be called
        repeatedly without restarting the process.
        """
        db_rules = await self._load_db_rules()
        converted = _convert_db_rules(db_rules)
        # Merge: injected rules first, then DB rules, sorted by priority
        all_rules = list(self._injected_rules) + converted
        all_rules.sort(key=lambda r: r.priority, reverse=True)
        self._injected_rules = all_rules
        logger.info("rules_reloaded", total=len(all_rules), from_db=len(converted))

    # ------------------------------------------------------------------
    # 3. detect  (main entry point)
    # ------------------------------------------------------------------

    async def detect(
        self,
        text: str,
        context: DetectionContext | dict | None = None,
    ) -> DetectionResult:
        """Run all rules against *text* and return aggregated results.

        1. Loads DB rules (legacy path)
        2. Evaluates injected Rule objects in parallel
        3. Merges both result sets
        4. Deduplicates contexts, orders by confidence
        """
        t0 = time.monotonic()

        # Normalise context
        if context is None:
            det_ctx = DetectionContext()
        elif isinstance(context, dict):
            det_ctx = DetectionContext(user_info=context)
        else:
            det_ctx = context

        triggered: list[TriggeredRule] = []
        topics: set[str] = set()

        # ── Legacy DB path (maintains backward compat) ────────────────
        db_rules = await self._load_db_rules()
        for rule in db_rules:
            config = rule.rule_config or {}

            if rule.rule_type == RuleType.KEYWORD:
                keywords = config.get("keywords", [])
                if keywords:
                    m = evaluate_keyword_rule(text, rule.id, rule.name, keywords)
                    if m.matched:
                        _append_legacy_match(triggered, m, rule, topics)

            elif rule.rule_type == RuleType.REGEX:
                pattern = config.get("pattern", "")
                if pattern:
                    m = evaluate_pattern_rule(text, rule.id, rule.name, pattern)
                    if m.matched:
                        _append_legacy_match(triggered, m, rule, topics)

            elif rule.rule_type == RuleType.COMPOSITE:
                keywords = config.get("keywords", [])
                pattern = config.get("pattern", "")
                if keywords:
                    m = evaluate_keyword_rule(text, rule.id, rule.name, keywords)
                    if m.matched:
                        _append_legacy_match(triggered, m, rule, topics)
                if pattern:
                    m = evaluate_pattern_rule(text, rule.id, rule.name, pattern)
                    if m.matched:
                        _append_legacy_match(triggered, m, rule, topics)

            elif rule.rule_type == RuleType.SEMANTIC:
                # Now handled by injected SemanticRule objects if loaded
                pass

        # ── New OOP path – parallel rule evaluation ───────────────────
        if self._injected_rules:
            rule_matches = await self._evaluate_rules_parallel(
                text, det_ctx,
            )
            for rm in rule_matches:
                triggered.append(TriggeredRule(
                    rule_id="",
                    rule_name=rm.rule_name,
                    confidence=rm.confidence,
                    matched_keywords=_extract_keywords(rm),
                ))
                topics.update(rm.contexts)

        overall_confidence = max(
            (t.confidence for t in triggered), default=0.0,
        )

        elapsed_ms = int((time.monotonic() - t0) * 1000)

        logger.info(
            "detection_complete",
            rules_checked=len(db_rules) + len(self._injected_rules),
            triggered=len(triggered),
            confidence=overall_confidence,
            elapsed_ms=elapsed_ms,
        )

        return DetectionResult(
            triggered_rules=triggered,
            suggested_topics=sorted(topics),
            confidence=overall_confidence,
            processing_time_ms=elapsed_ms,
        )

    # ------------------------------------------------------------------
    # 4. detect_and_enrich
    # ------------------------------------------------------------------

    async def detect_and_enrich(
        self,
        messages: list[dict],
        context: DetectionContext | None = None,
        knowledge_svc: KnowledgeService | None = None,
    ) -> EnrichedRequest:
        """Detect on the last user message, then inject RAG context.

        Parameters
        ----------
        messages : list[dict]
            Chat messages in ``[{"role": ..., "content": ...}]`` format.
        context : DetectionContext | None
            Optional detection context.
        knowledge_svc : KnowledgeService | None
            If provided, used to build RAG context.
        """
        # Extract last user message
        user_text = ""
        for msg in reversed(messages):
            if msg.get("role") == "user":
                user_text = msg.get("content", "")
                break

        detection = await self.detect(user_text, context)

        rag_context = ""
        enriched = list(messages)

        if detection.confidence >= 0.3 and detection.suggested_topics and knowledge_svc:
            rag_context = await knowledge_svc.build_rag_context(
                query=user_text,
                detected_contexts=detection.suggested_topics,
            )

            if rag_context:
                rag_system = RAG_SYSTEM_TEMPLATE.format(context=rag_context)
                enriched = _inject_system_message(messages, rag_system)

        return EnrichedRequest(
            original_messages=messages,
            enriched_messages=enriched,
            rag_context=rag_context,
            detected=detection,
        )

    # ------------------------------------------------------------------
    # Parallel rule evaluation
    # ------------------------------------------------------------------

    async def _evaluate_rules_parallel(
        self,
        text: str,
        context: DetectionContext,
    ) -> list[RuleMatch]:
        """Evaluate all injected rules concurrently with per-rule timeout."""

        async def _safe_eval(rule: Rule) -> RuleMatch | None:
            try:
                return await asyncio.wait_for(
                    rule.match(text, context),
                    timeout=self._rule_timeout,
                )
            except asyncio.TimeoutError:
                logger.warning("rule_timeout", rule=rule.name, timeout=self._rule_timeout)
                return None
            except Exception as exc:
                logger.warning("rule_error", rule=rule.name, error=str(exc))
                return None

        results = await asyncio.gather(
            *(_safe_eval(r) for r in self._injected_rules),
        )

        return [r for r in results if r is not None]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _append_legacy_match(
    triggered: list[TriggeredRule],
    match: _LegacyRuleMatch,
    rule: DetectionRule,
    topics: set[str],
) -> None:
    """Add a legacy match result to the triggered list."""
    triggered.append(
        TriggeredRule(
            rule_id=match.rule_id,
            rule_name=match.rule_name,
            confidence=match.confidence,
            matched_keywords=match.matched_keywords,
        )
    )
    topics.update(rule.target_contexts or [])


def _extract_keywords(rm: RuleMatch) -> list[str]:
    """Pull keyword-like values from a RuleMatch's extracted dict."""
    kws = rm.extracted.get("matched_keywords", [])
    if kws:
        return kws
    matches = rm.extracted.get("matches", [])
    if matches:
        return matches
    return []


def _inject_system_message(messages: list[dict], system_content: str) -> list[dict]:
    """Insert a system message right after the first existing system message."""
    enriched: list[dict] = []
    injected = False
    for msg in messages:
        enriched.append(msg)
        if msg["role"] == "system" and not injected:
            enriched.append({"role": "system", "content": system_content})
            injected = True
    if not injected:
        enriched.insert(0, {"role": "system", "content": system_content})
    return enriched


def _convert_db_rules(db_rules: list[DetectionRule]) -> list[Rule]:
    """Convert DB DetectionRule rows to Rule objects."""
    converted: list[Rule] = []
    for db_rule in db_rules:
        config = db_rule.rule_config or {}
        contexts = db_rule.target_contexts or []
        priority = db_rule.priority or 0

        if db_rule.rule_type == RuleType.KEYWORD:
            keywords = config.get("keywords", [])
            if keywords:
                converted.append(KeywordRule(
                    name=db_rule.name,
                    keywords=keywords,
                    priority=priority,
                    contexts=contexts,
                    case_sensitive=config.get("case_sensitive", False),
                    match_all=config.get("match_all", False),
                ))

        elif db_rule.rule_type == RuleType.REGEX:
            patterns = config.get("patterns", [])
            pattern = config.get("pattern", "")
            if pattern and not patterns:
                patterns = [pattern]
            if patterns:
                converted.append(RegexRule(
                    name=db_rule.name,
                    patterns=patterns,
                    priority=priority,
                    contexts=contexts,
                ))

        elif db_rule.rule_type == RuleType.COMPOSITE:
            # Build child rules from config
            children: list[Rule] = []
            keywords = config.get("keywords", [])
            pattern = config.get("pattern", "")
            if keywords:
                children.append(KeywordRule(
                    name=f"{db_rule.name}_kw",
                    keywords=keywords,
                    contexts=contexts,
                ))
            if pattern:
                children.append(RegexRule(
                    name=f"{db_rule.name}_re",
                    patterns=[pattern],
                    contexts=contexts,
                ))
            if children:
                converted.append(CompositeRule(
                    name=db_rule.name,
                    rules=children,
                    operator=config.get("operator", "OR"),
                    priority=priority,
                    contexts=contexts,
                ))

        elif db_rule.rule_type == RuleType.SEMANTIC:
            ref_texts = config.get("reference_texts", [])
            threshold = config.get("threshold", 0.75)
            if ref_texts:
                converted.append(SemanticRule(
                    name=db_rule.name,
                    reference_texts=ref_texts,
                    threshold=threshold,
                    priority=priority,
                    contexts=contexts,
                ))

    return converted
