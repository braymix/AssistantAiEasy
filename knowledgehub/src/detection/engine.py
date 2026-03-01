"""Context detection engine – evaluates rules against input text."""

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from src.config.logging import get_logger
from src.detection.rules import evaluate_keyword_rule, evaluate_pattern_rule
from src.gateway.schemas.detection import DetectionResult, TriggeredRule
from src.knowledge.models import DetectionRule

logger = get_logger(__name__)


class DetectionEngine:
    def __init__(self, session: AsyncSession):
        self._session = session

    async def _load_rules(self) -> list[DetectionRule]:
        q = select(DetectionRule).where(DetectionRule.enabled.is_(True)).order_by(DetectionRule.priority.desc())
        result = await self._session.execute(q)
        return list(result.scalars().all())

    async def detect(self, text: str, context: dict | None = None) -> DetectionResult:
        """Run all enabled rules against the input text."""
        rules = await self._load_rules()
        triggered: list[TriggeredRule] = []
        topics: set[str] = set()

        for rule in rules:
            # Keyword-based detection
            keywords = rule.keywords or []
            if keywords:
                match = evaluate_keyword_rule(text, rule.id, rule.name, keywords)
                if match.matched:
                    triggered.append(
                        TriggeredRule(
                            rule_id=match.rule_id,
                            rule_name=match.rule_name,
                            confidence=match.confidence,
                            matched_keywords=match.matched_keywords,
                        )
                    )
                    topics.add(rule.name)

            # Pattern-based detection
            if rule.pattern:
                match = evaluate_pattern_rule(text, rule.id, rule.name, rule.pattern)
                if match.matched:
                    triggered.append(
                        TriggeredRule(
                            rule_id=match.rule_id,
                            rule_name=match.rule_name,
                            confidence=match.confidence,
                            matched_keywords=match.matched_keywords,
                        )
                    )
                    topics.add(rule.name)

        overall_confidence = max((t.confidence for t in triggered), default=0.0)

        logger.info(
            "detection_complete",
            rules_checked=len(rules),
            triggered=len(triggered),
            confidence=overall_confidence,
        )

        return DetectionResult(
            triggered_rules=triggered,
            suggested_topics=sorted(topics),
            confidence=overall_confidence,
        )
