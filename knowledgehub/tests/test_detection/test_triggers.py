"""Tests for trigger evaluation."""

from src.detection.triggers import TriggerAction, Trigger, get_applicable_triggers


def test_high_confidence_triggers_all():
    triggers = get_applicable_triggers(confidence=0.9, topics=["general"])
    names = [t.name for t in triggers]
    assert "knowledge_enrichment" in names
    assert "high_confidence_alert" in names
    assert "analytics_logger" in names


def test_low_confidence_excludes_alert():
    triggers = get_applicable_triggers(confidence=0.2, topics=["general"])
    names = [t.name for t in triggers]
    assert "high_confidence_alert" not in names
    assert "analytics_logger" in names


def test_topic_filter():
    custom_triggers = [
        Trigger(name="specific", action=TriggerAction.ENRICH, min_confidence=0.0, target_topics=["security"]),
    ]
    result = get_applicable_triggers(confidence=0.5, topics=["general"], triggers=custom_triggers)
    assert len(result) == 0

    result = get_applicable_triggers(confidence=0.5, topics=["security"], triggers=custom_triggers)
    assert len(result) == 1
