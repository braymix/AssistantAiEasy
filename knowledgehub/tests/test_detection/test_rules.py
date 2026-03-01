"""Tests for detection rule evaluation."""

from src.detection.rules import evaluate_keyword_rule, evaluate_pattern_rule


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
