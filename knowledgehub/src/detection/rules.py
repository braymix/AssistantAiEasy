"""
Detection rules – abstract base and concrete implementations.

Rule types:
  1. KeywordRule     – substring matching (AND / OR, case sensitivity)
  2. RegexRule       – compiled regex with named groups
  3. SemanticRule    – embedding-based similarity
  4. CompositeRule   – AND / OR / NOT combination of child rules
  5. LLMRule         – LLM-based classification with result caching
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import math
import re
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from src.config.logging import get_logger

if TYPE_CHECKING:
    from src.knowledge.embeddings import EmbeddingProvider
    from src.llm.base import LLMProvider

logger = get_logger(__name__)

# Default timeout for individual rule evaluation (seconds)
_RULE_TIMEOUT = 5.0


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class DetectionContext:
    """Contextual information passed to every rule during evaluation."""

    conversation_history: list[dict] = field(default_factory=list)
    user_info: dict = field(default_factory=dict)


@dataclass
class RuleMatch:
    """Result of a single rule evaluation that matched."""

    rule_name: str
    confidence: float
    extracted: dict = field(default_factory=dict)
    contexts: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Abstract base
# ---------------------------------------------------------------------------


class Rule(ABC):
    """Base class for all detection rules."""

    def __init__(self, name: str, priority: int = 0, contexts: list[str] | None = None):
        self.name = name
        self.priority = priority
        self.contexts = contexts or []

    @abstractmethod
    async def match(self, text: str, context: DetectionContext) -> RuleMatch | None:
        """Evaluate the rule against *text*.

        Returns a :class:`RuleMatch` when the rule fires, ``None`` otherwise.
        """


# ═══════════════════════════════════════════════════════════════════════════
# 1. KeywordRule
# ═══════════════════════════════════════════════════════════════════════════


class KeywordRule(Rule):
    """Match when keywords are found in the input text.

    Parameters
    ----------
    keywords : list[str]
        Words or phrases to search for.
    case_sensitive : bool
        If ``False`` (default), comparison is case-insensitive.
    match_all : bool
        If ``True``, *all* keywords must be present (AND).
        If ``False`` (default), *any* keyword suffices (OR).
    """

    def __init__(
        self,
        name: str,
        keywords: list[str],
        *,
        priority: int = 0,
        contexts: list[str] | None = None,
        case_sensitive: bool = False,
        match_all: bool = False,
    ):
        super().__init__(name, priority, contexts)
        self.keywords = keywords
        self.case_sensitive = case_sensitive
        self.match_all = match_all

    async def match(self, text: str, context: DetectionContext) -> RuleMatch | None:
        compare_text = text if self.case_sensitive else text.lower()
        matched: list[str] = []
        for kw in self.keywords:
            compare_kw = kw if self.case_sensitive else kw.lower()
            if compare_kw in compare_text:
                matched.append(kw)

        if not matched:
            return None

        if self.match_all and len(matched) < len(self.keywords):
            return None

        confidence = min(len(matched) / max(len(self.keywords), 1), 1.0)
        return RuleMatch(
            rule_name=self.name,
            confidence=confidence,
            extracted={"matched_keywords": matched},
            contexts=self.contexts,
        )


# ═══════════════════════════════════════════════════════════════════════════
# 2. RegexRule
# ═══════════════════════════════════════════════════════════════════════════


class RegexRule(Rule):
    """Match when one or more regex patterns hit.

    Patterns are compiled once at construction time for efficiency.
    """

    def __init__(
        self,
        name: str,
        patterns: list[str],
        *,
        priority: int = 0,
        contexts: list[str] | None = None,
    ):
        super().__init__(name, priority, contexts)
        self._compiled: list[re.Pattern[str]] = []
        for pat in patterns:
            try:
                self._compiled.append(re.compile(pat, re.IGNORECASE))
            except re.error as exc:
                logger.warning("regex_compile_error", pattern=pat, error=str(exc))

    async def match(self, text: str, context: DetectionContext) -> RuleMatch | None:
        all_matches: list[str] = []
        all_groups: dict[str, str] = {}

        for regex in self._compiled:
            m = regex.search(text)
            if m:
                all_matches.append(m.group())
                all_groups.update({k: v for k, v in m.groupdict().items() if v is not None})

        if not all_matches:
            return None

        return RuleMatch(
            rule_name=self.name,
            confidence=0.8,
            extracted={"matches": all_matches, "groups": all_groups},
            contexts=self.contexts,
        )


# ═══════════════════════════════════════════════════════════════════════════
# 3. SemanticRule
# ═══════════════════════════════════════════════════════════════════════════


class SemanticRule(Rule):
    """Match when input is semantically similar to reference texts.

    Uses the configured :class:`EmbeddingProvider` to compute cosine
    similarity.  Reference embeddings are computed lazily and cached.
    """

    def __init__(
        self,
        name: str,
        reference_texts: list[str],
        threshold: float = 0.75,
        *,
        priority: int = 0,
        contexts: list[str] | None = None,
        embedder: EmbeddingProvider | None = None,
    ):
        super().__init__(name, priority, contexts)
        self.reference_texts = reference_texts
        self.threshold = threshold
        self._embedder = embedder
        self._reference_embeddings: list[list[float]] | None = None

    def _get_embedder(self) -> EmbeddingProvider:
        if self._embedder is None:
            from src.knowledge.embeddings import get_embedding_provider
            self._embedder = get_embedding_provider()
        return self._embedder

    async def _ensure_reference_embeddings(self) -> list[list[float]]:
        if self._reference_embeddings is None:
            embedder = self._get_embedder()
            self._reference_embeddings = await embedder.embed(self.reference_texts)
        return self._reference_embeddings

    async def match(self, text: str, context: DetectionContext) -> RuleMatch | None:
        embedder = self._get_embedder()
        ref_embeddings = await self._ensure_reference_embeddings()
        query_embedding = await embedder.embed_single(text)

        best_score = 0.0
        best_ref = ""
        for ref_text, ref_emb in zip(self.reference_texts, ref_embeddings):
            score = _cosine_similarity(query_embedding, ref_emb)
            if score > best_score:
                best_score = score
                best_ref = ref_text

        if best_score < self.threshold:
            return None

        return RuleMatch(
            rule_name=self.name,
            confidence=best_score,
            extracted={"best_match": best_ref, "similarity": best_score},
            contexts=self.contexts,
        )


# ═══════════════════════════════════════════════════════════════════════════
# 4. CompositeRule
# ═══════════════════════════════════════════════════════════════════════════


class CompositeRule(Rule):
    """Combine child rules with AND / OR / NOT operators.

    - ``AND``  – all child rules must match
    - ``OR``   – at least one child rule must match
    - ``NOT``  – none of the child rules should match
    """

    def __init__(
        self,
        name: str,
        rules: list[Rule],
        operator: str = "AND",
        *,
        priority: int = 0,
        contexts: list[str] | None = None,
    ):
        super().__init__(name, priority, contexts)
        if operator.upper() not in ("AND", "OR", "NOT"):
            raise ValueError(f"Invalid CompositeRule operator: {operator!r}")
        self.rules = rules
        self.operator = operator.upper()

    async def match(self, text: str, context: DetectionContext) -> RuleMatch | None:
        # Evaluate all child rules in parallel
        results = await asyncio.gather(
            *(r.match(text, context) for r in self.rules),
            return_exceptions=True,
        )

        child_matches: list[RuleMatch] = []
        for res in results:
            if isinstance(res, Exception):
                logger.warning("composite_child_error", rule=self.name, error=str(res))
                continue
            if res is not None:
                child_matches.append(res)

        if self.operator == "AND":
            if len(child_matches) < len(self.rules):
                return None
            avg_confidence = sum(m.confidence for m in child_matches) / len(child_matches)
            merged_extracted = _merge_extracted(child_matches)
            merged_contexts = _merge_contexts(child_matches, self.contexts)
            return RuleMatch(
                rule_name=self.name,
                confidence=avg_confidence,
                extracted=merged_extracted,
                contexts=merged_contexts,
            )

        elif self.operator == "OR":
            if not child_matches:
                return None
            best = max(child_matches, key=lambda m: m.confidence)
            merged_extracted = _merge_extracted(child_matches)
            merged_contexts = _merge_contexts(child_matches, self.contexts)
            return RuleMatch(
                rule_name=self.name,
                confidence=best.confidence,
                extracted=merged_extracted,
                contexts=merged_contexts,
            )

        else:  # NOT
            if child_matches:
                return None
            return RuleMatch(
                rule_name=self.name,
                confidence=1.0,
                extracted={},
                contexts=self.contexts,
            )


# ═══════════════════════════════════════════════════════════════════════════
# 5. LLMRule
# ═══════════════════════════════════════════════════════════════════════════


class LLMRule(Rule):
    """Ask the LLM whether the input matches a described pattern.

    Results are cached by content hash for efficiency.
    """

    def __init__(
        self,
        name: str,
        prompt_template: str,
        *,
        priority: int = 0,
        contexts: list[str] | None = None,
        llm: LLMProvider | None = None,
        cache_maxsize: int = 256,
    ):
        super().__init__(name, priority, contexts)
        self.prompt_template = prompt_template
        self._llm = llm
        self._cache: dict[str, RuleMatch | None] = {}
        self._cache_maxsize = cache_maxsize

    def _get_llm(self) -> LLMProvider:
        if self._llm is None:
            from src.llm.base import get_llm_provider
            self._llm = get_llm_provider()
        return self._llm

    async def match(self, text: str, context: DetectionContext) -> RuleMatch | None:
        cache_key = hashlib.md5(text.encode()).hexdigest()

        if cache_key in self._cache:
            return self._cache[cache_key]

        llm = self._get_llm()
        prompt = self.prompt_template.format(text=text)

        try:
            response = await llm.chat(
                [
                    {
                        "role": "system",
                        "content": (
                            "You are a classification assistant. "
                            "Respond with a JSON object: "
                            '{"match": true/false, "confidence": 0.0-1.0, "reason": "..."}'
                        ),
                    },
                    {"role": "user", "content": prompt},
                ],
                temperature=0.0,
            )
        except Exception as exc:
            logger.warning("llm_rule_error", rule=self.name, error=str(exc))
            return None

        result = _parse_llm_classification(response, self.name, self.contexts)

        # Cache management
        if len(self._cache) >= self._cache_maxsize:
            oldest = next(iter(self._cache))
            del self._cache[oldest]
        self._cache[cache_key] = result

        return result


# ---------------------------------------------------------------------------
# Legacy evaluation functions (backward compatibility with existing tests)
# ---------------------------------------------------------------------------

@dataclass
class _LegacyRuleMatch:
    """Legacy rule-match used by the old engine path and existing tests."""

    rule_id: str
    rule_name: str
    matched: bool
    confidence: float = 0.0
    matched_keywords: list[str] = field(default_factory=list)


def evaluate_keyword_rule(
    text: str,
    rule_id: str,
    rule_name: str,
    keywords: list[str],
) -> _LegacyRuleMatch:
    """Check if any keywords from the rule appear in the text."""
    text_lower = text.lower()
    matched_keywords = [kw for kw in keywords if kw.lower() in text_lower]

    if not matched_keywords:
        return _LegacyRuleMatch(rule_id=rule_id, rule_name=rule_name, matched=False)

    confidence = min(len(matched_keywords) / max(len(keywords), 1), 1.0)
    return _LegacyRuleMatch(
        rule_id=rule_id,
        rule_name=rule_name,
        matched=True,
        confidence=confidence,
        matched_keywords=matched_keywords,
    )


def evaluate_pattern_rule(
    text: str,
    rule_id: str,
    rule_name: str,
    pattern: str,
) -> _LegacyRuleMatch:
    """Check if a regex pattern matches the text."""
    try:
        m = re.search(pattern, text, re.IGNORECASE)
    except re.error:
        return _LegacyRuleMatch(rule_id=rule_id, rule_name=rule_name, matched=False)

    if m:
        return _LegacyRuleMatch(
            rule_id=rule_id,
            rule_name=rule_name,
            matched=True,
            confidence=0.8,
            matched_keywords=[m.group()],
        )
    return _LegacyRuleMatch(rule_id=rule_id, rule_name=rule_name, matched=False)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _cosine_similarity(a: list[float], b: list[float]) -> float:
    """Compute cosine similarity between two vectors."""
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(x * x for x in b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


def _merge_extracted(matches: list[RuleMatch]) -> dict:
    """Merge ``extracted`` dicts from multiple child matches."""
    merged: dict[str, Any] = {}
    for m in matches:
        for k, v in m.extracted.items():
            if k in merged:
                existing = merged[k]
                if isinstance(existing, list) and isinstance(v, list):
                    merged[k] = existing + v
                else:
                    merged[k] = v
            else:
                merged[k] = v
    return merged


def _merge_contexts(matches: list[RuleMatch], own_contexts: list[str]) -> list[str]:
    """Merge contexts from children + the composite rule's own contexts."""
    seen: set[str] = set()
    result: list[str] = []
    for ctx in own_contexts:
        if ctx not in seen:
            result.append(ctx)
            seen.add(ctx)
    for m in matches:
        for ctx in m.contexts:
            if ctx not in seen:
                result.append(ctx)
                seen.add(ctx)
    return result


def _parse_llm_classification(
    response: str, rule_name: str, contexts: list[str],
) -> RuleMatch | None:
    """Parse the LLM JSON response for classification."""
    text = response.strip()

    # Try extracting JSON from code blocks
    if "```" in text:
        for block in text.split("```"):
            block = block.strip()
            if block.startswith("json"):
                block = block[4:].strip()
            try:
                data = json.loads(block)
                if isinstance(data, dict):
                    if data.get("match"):
                        return RuleMatch(
                            rule_name=rule_name,
                            confidence=float(data.get("confidence", 0.8)),
                            extracted={"reason": data.get("reason", "")},
                            contexts=contexts,
                        )
                    return None
            except (json.JSONDecodeError, ValueError):
                continue

    # Try direct JSON parse
    try:
        data = json.loads(text)
        if isinstance(data, dict):
            if data.get("match"):
                return RuleMatch(
                    rule_name=rule_name,
                    confidence=float(data.get("confidence", 0.8)),
                    extracted={"reason": data.get("reason", "")},
                    contexts=contexts,
                )
            return None
    except (json.JSONDecodeError, ValueError):
        pass

    # Try to find JSON object in text
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        try:
            data = json.loads(text[start : end + 1])
            if isinstance(data, dict) and data.get("match"):
                return RuleMatch(
                    rule_name=rule_name,
                    confidence=float(data.get("confidence", 0.8)),
                    extracted={"reason": data.get("reason", "")},
                    contexts=contexts,
                )
            return None
        except (json.JSONDecodeError, ValueError):
            pass

    logger.warning("llm_rule_parse_error", rule=rule_name, response=text[:200])
    return None
