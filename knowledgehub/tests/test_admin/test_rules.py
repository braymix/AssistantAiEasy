"""Tests for the Admin Rules REST API."""

from __future__ import annotations

import pytest


BASE = "/api/v1/admin"


# ═══════════════════════════════════════════════════════════════════════════
# Auth
# ═══════════════════════════════════════════════════════════════════════════


@pytest.mark.asyncio
async def test_missing_api_key_returns_401(admin_client):
    resp = await admin_client.get(
        f"{BASE}/rules",
        headers={"X-API-Key": ""},
    )
    assert resp.status_code == 401


@pytest.mark.asyncio
async def test_wrong_api_key_returns_401(admin_client):
    resp = await admin_client.get(
        f"{BASE}/rules",
        headers={"X-API-Key": "wrong-key"},
    )
    assert resp.status_code == 401


# ═══════════════════════════════════════════════════════════════════════════
# LIST
# ═══════════════════════════════════════════════════════════════════════════


@pytest.mark.asyncio
async def test_list_rules_empty(admin_client):
    resp = await admin_client.get(f"{BASE}/rules")
    assert resp.status_code == 200
    body = resp.json()
    assert body["data"] == []
    assert body["meta"]["total"] == 0


# ═══════════════════════════════════════════════════════════════════════════
# CREATE
# ═══════════════════════════════════════════════════════════════════════════


@pytest.mark.asyncio
async def test_create_rule(admin_client):
    resp = await admin_client.post(f"{BASE}/rules", json={
        "name": "test_kw_rule",
        "rule_type": "keyword",
        "rule_config": {"keywords": ["python", "fastapi"]},
        "target_contexts": ["tech"],
        "priority": 5,
    })
    assert resp.status_code == 201
    data = resp.json()["data"]
    assert data["name"] == "test_kw_rule"
    assert data["rule_type"] == "keyword"
    assert data["priority"] == 5
    assert data["enabled"] is True
    assert "id" in data


@pytest.mark.asyncio
async def test_create_rule_invalid_type(admin_client):
    resp = await admin_client.post(f"{BASE}/rules", json={
        "name": "bad_rule",
        "rule_type": "invalid_type",
    })
    assert resp.status_code == 422


# ═══════════════════════════════════════════════════════════════════════════
# GET
# ═══════════════════════════════════════════════════════════════════════════


@pytest.mark.asyncio
async def test_get_rule(admin_client):
    create_resp = await admin_client.post(f"{BASE}/rules", json={
        "name": "get_test_rule",
        "rule_type": "regex",
        "rule_config": {"patterns": ["\\d+"]},
    })
    rule_id = create_resp.json()["data"]["id"]

    resp = await admin_client.get(f"{BASE}/rules/{rule_id}")
    assert resp.status_code == 200
    assert resp.json()["data"]["name"] == "get_test_rule"


@pytest.mark.asyncio
async def test_get_rule_not_found(admin_client):
    resp = await admin_client.get(f"{BASE}/rules/nonexistent-id")
    assert resp.status_code == 404


# ═══════════════════════════════════════════════════════════════════════════
# UPDATE
# ═══════════════════════════════════════════════════════════════════════════


@pytest.mark.asyncio
async def test_update_rule(admin_client):
    create_resp = await admin_client.post(f"{BASE}/rules", json={
        "name": "update_me",
        "rule_type": "keyword",
        "rule_config": {"keywords": ["old"]},
        "priority": 1,
    })
    rule_id = create_resp.json()["data"]["id"]

    resp = await admin_client.put(f"{BASE}/rules/{rule_id}", json={
        "name": "updated_name",
        "priority": 10,
        "enabled": False,
    })
    assert resp.status_code == 200
    data = resp.json()["data"]
    assert data["name"] == "updated_name"
    assert data["priority"] == 10
    assert data["enabled"] is False


@pytest.mark.asyncio
async def test_update_rule_not_found(admin_client):
    resp = await admin_client.put(f"{BASE}/rules/missing", json={"name": "x"})
    assert resp.status_code == 404


# ═══════════════════════════════════════════════════════════════════════════
# DELETE
# ═══════════════════════════════════════════════════════════════════════════


@pytest.mark.asyncio
async def test_delete_rule(admin_client):
    create_resp = await admin_client.post(f"{BASE}/rules", json={
        "name": "delete_me",
        "rule_type": "keyword",
        "rule_config": {"keywords": ["bye"]},
    })
    rule_id = create_resp.json()["data"]["id"]

    del_resp = await admin_client.delete(f"{BASE}/rules/{rule_id}")
    assert del_resp.status_code == 204

    get_resp = await admin_client.get(f"{BASE}/rules/{rule_id}")
    assert get_resp.status_code == 404


@pytest.mark.asyncio
async def test_delete_rule_not_found(admin_client):
    resp = await admin_client.delete(f"{BASE}/rules/no-such-id")
    assert resp.status_code == 404


# ═══════════════════════════════════════════════════════════════════════════
# LIST with filters
# ═══════════════════════════════════════════════════════════════════════════


@pytest.mark.asyncio
async def test_list_rules_with_filters(admin_client):
    # Create 2 rules
    await admin_client.post(f"{BASE}/rules", json={
        "name": "filter_kw", "rule_type": "keyword",
        "rule_config": {"keywords": ["a"]}, "enabled": True,
    })
    await admin_client.post(f"{BASE}/rules", json={
        "name": "filter_regex", "rule_type": "regex",
        "rule_config": {"patterns": ["b"]}, "enabled": False,
    })

    # Filter by type
    resp = await admin_client.get(f"{BASE}/rules", params={"rule_type": "keyword"})
    assert resp.status_code == 200
    assert all(r["rule_type"] == "keyword" for r in resp.json()["data"])

    # Filter by enabled
    resp = await admin_client.get(f"{BASE}/rules", params={"enabled": False})
    assert all(r["enabled"] is False for r in resp.json()["data"])

    # Search by name
    resp = await admin_client.get(f"{BASE}/rules", params={"search": "filter_kw"})
    assert len(resp.json()["data"]) >= 1
    assert resp.json()["data"][0]["name"] == "filter_kw"


# ═══════════════════════════════════════════════════════════════════════════
# TEST endpoint
# ═══════════════════════════════════════════════════════════════════════════


@pytest.mark.asyncio
async def test_test_rule_match(admin_client):
    create_resp = await admin_client.post(f"{BASE}/rules", json={
        "name": "test_keyword",
        "rule_type": "keyword",
        "rule_config": {"keywords": ["python"]},
        "target_contexts": ["tech"],
    })
    rule_id = create_resp.json()["data"]["id"]

    resp = await admin_client.post(f"{BASE}/rules/{rule_id}/test", json={
        "text": "I love python programming",
    })
    assert resp.status_code == 200
    result = resp.json()["data"]
    assert result["matched"] is True
    assert result["confidence"] > 0


@pytest.mark.asyncio
async def test_test_rule_no_match(admin_client):
    create_resp = await admin_client.post(f"{BASE}/rules", json={
        "name": "test_keyword_no",
        "rule_type": "keyword",
        "rule_config": {"keywords": ["rust"]},
    })
    rule_id = create_resp.json()["data"]["id"]

    resp = await admin_client.post(f"{BASE}/rules/{rule_id}/test", json={
        "text": "I love python",
    })
    assert resp.status_code == 200
    assert resp.json()["data"]["matched"] is False


@pytest.mark.asyncio
async def test_test_rule_not_found(admin_client):
    resp = await admin_client.post(f"{BASE}/rules/missing/test", json={"text": "hello"})
    assert resp.status_code == 404


# ═══════════════════════════════════════════════════════════════════════════
# RELOAD
# ═══════════════════════════════════════════════════════════════════════════


@pytest.mark.asyncio
async def test_reload_rules(admin_client):
    resp = await admin_client.post(f"{BASE}/rules/reload")
    assert resp.status_code == 200
    data = resp.json()["data"]
    assert data["reloaded"] is True
    assert isinstance(data["rule_count"], int)


# ═══════════════════════════════════════════════════════════════════════════
# PAGINATION
# ═══════════════════════════════════════════════════════════════════════════


@pytest.mark.asyncio
async def test_list_rules_pagination(admin_client):
    # Create 3 rules
    for i in range(3):
        await admin_client.post(f"{BASE}/rules", json={
            "name": f"page_rule_{i}", "rule_type": "keyword",
            "rule_config": {"keywords": [f"kw{i}"]}, "priority": i,
        })

    resp = await admin_client.get(f"{BASE}/rules", params={"limit": 2, "offset": 0})
    body = resp.json()
    assert len(body["data"]) == 2
    assert body["meta"]["total"] >= 3
    assert body["meta"]["limit"] == 2
    assert body["meta"]["offset"] == 0

    resp2 = await admin_client.get(f"{BASE}/rules", params={"limit": 2, "offset": 2})
    assert len(resp2.json()["data"]) >= 1
