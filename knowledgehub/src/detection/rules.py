"""Detection rule evaluation logic."""

import re
from dataclasses import dataclass, field


@dataclass
class RuleMatch:
    """Result of evaluating a single detection rule against input text."""

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
) -> RuleMatch:
    """Check if any keywords from the rule appear in the text."""
    text_lower = text.lower()
    matched_keywords = [kw for kw in keywords if kw.lower() in text_lower]

    if not matched_keywords:
        return RuleMatch(rule_id=rule_id, rule_name=rule_name, matched=False)

    # Confidence scales with proportion of keywords matched
    confidence = min(len(matched_keywords) / max(len(keywords), 1), 1.0)
    return RuleMatch(
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
) -> RuleMatch:
    """Check if a regex pattern matches the text."""
    try:
        match = re.search(pattern, text, re.IGNORECASE)
    except re.error:
        return RuleMatch(rule_id=rule_id, rule_name=rule_name, matched=False)

    if match:
        return RuleMatch(
            rule_id=rule_id,
            rule_name=rule_name,
            matched=True,
            confidence=0.8,
            matched_keywords=[match.group()],
        )
    return RuleMatch(rule_id=rule_id, rule_name=rule_name, matched=False)
