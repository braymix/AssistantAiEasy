"""Tests for the Admin REST API.

Covers:
  - CRUD operations for rules, contexts, and knowledge
  - Analytics endpoints
  - Authentication
"""

from __future__ import annotations

import pytest

from src.shared.models import ContentType, Context, DetectionRule, KnowledgeItem, RuleType


BASE = "/api/v1/admin"


# ═══════════════════════════════════════════════════════════════════════════════
# 1. CRUD Rules
# ═══════════════════════════════════════════════════════════════════════════════


@pytest.mark.asyncio
async def test_crud_rules_create(admin_client):
    """Create a new detection rule via the API."""
    resp = await admin_client.post(f"{BASE}/rules", json={
        "name": "test_rule",
        "description": "Test rule for API",
        "rule_type": "keyword",
        "rule_config": {"keywords": ["test", "example"]},
        "target_contexts": ["testing"],
        "priority": 5,
        "enabled": True,
    })
    assert resp.status_code == 201
    data = resp.json()["data"]
    assert data["name"] == "test_rule"
    assert data["rule_type"] == "keyword"
    return data["id"]


@pytest.mark.asyncio
async def test_crud_rules_list(admin_client):
    """List rules returns paginated data."""
    # Create a rule first
    await admin_client.post(f"{BASE}/rules", json={
        "name": "list_test_rule",
        "rule_type": "keyword",
        "rule_config": {"keywords": ["list"]},
        "target_contexts": ["test"],
    })

    resp = await admin_client.get(f"{BASE}/rules")
    assert resp.status_code == 200
    body = resp.json()
    assert "data" in body
    assert "meta" in body
    assert body["meta"]["total"] >= 1


@pytest.mark.asyncio
async def test_crud_rules_get_update_delete(admin_client):
    """Full CRUD cycle: create → get → update → delete."""
    # Create
    create_resp = await admin_client.post(f"{BASE}/rules", json={
        "name": "crud_cycle_rule",
        "rule_type": "regex",
        "rule_config": {"pattern": r"ERR-\d+"},
        "target_contexts": ["errors"],
        "priority": 3,
    })
    assert create_resp.status_code == 201
    rule_id = create_resp.json()["data"]["id"]

    # Get
    get_resp = await admin_client.get(f"{BASE}/rules/{rule_id}")
    assert get_resp.status_code == 200
    assert get_resp.json()["data"]["name"] == "crud_cycle_rule"

    # Update
    update_resp = await admin_client.put(f"{BASE}/rules/{rule_id}", json={
        "name": "crud_cycle_rule_updated",
        "priority": 10,
    })
    assert update_resp.status_code == 200
    assert update_resp.json()["data"]["name"] == "crud_cycle_rule_updated"
    assert update_resp.json()["data"]["priority"] == 10

    # Delete
    del_resp = await admin_client.delete(f"{BASE}/rules/{rule_id}")
    assert del_resp.status_code == 204

    # Verify deleted
    get_after = await admin_client.get(f"{BASE}/rules/{rule_id}")
    assert get_after.status_code == 404


# ═══════════════════════════════════════════════════════════════════════════════
# 2. CRUD Contexts
# ═══════════════════════════════════════════════════════════════════════════════


@pytest.mark.asyncio
async def test_crud_contexts_create(admin_client):
    """Create a new context."""
    resp = await admin_client.post(f"{BASE}/contexts", json={
        "name": "test_context",
        "description": "A test context for the API",
    })
    assert resp.status_code == 201
    data = resp.json()["data"]
    assert data["name"] == "test_context"


@pytest.mark.asyncio
async def test_crud_contexts_list(admin_client):
    """List contexts returns all contexts."""
    await admin_client.post(f"{BASE}/contexts", json={
        "name": "list_ctx",
        "description": "For listing",
    })

    resp = await admin_client.get(f"{BASE}/contexts")
    assert resp.status_code == 200
    assert len(resp.json()["data"]) >= 1


@pytest.mark.asyncio
async def test_crud_contexts_update_delete(admin_client):
    """Create → update → delete context."""
    # Create
    create_resp = await admin_client.post(f"{BASE}/contexts", json={
        "name": "ctx_to_modify",
        "description": "Original description",
    })
    ctx_id = create_resp.json()["data"]["id"]

    # Update
    update_resp = await admin_client.put(f"{BASE}/contexts/{ctx_id}", json={
        "description": "Updated description",
    })
    assert update_resp.status_code == 200
    assert update_resp.json()["data"]["description"] == "Updated description"

    # Delete
    del_resp = await admin_client.delete(f"{BASE}/contexts/{ctx_id}")
    assert del_resp.status_code == 204


# ═══════════════════════════════════════════════════════════════════════════════
# 3. Knowledge management
# ═══════════════════════════════════════════════════════════════════════════════


@pytest.mark.asyncio
async def test_knowledge_management_list(admin_client, db_session):
    """List knowledge items returns paginated results."""
    # Seed a knowledge item directly
    item = KnowledgeItem(
        content="Test knowledge for API",
        content_type=ContentType.MANUAL,
        contexts=["test"],
        verified=False,
        created_by="test",
    )
    db_session.add(item)
    await db_session.flush()

    resp = await admin_client.get(f"{BASE}/knowledge")
    assert resp.status_code == 200
    body = resp.json()
    assert "data" in body


@pytest.mark.asyncio
async def test_knowledge_management_pending(admin_client, db_session):
    """List pending (unverified) knowledge items."""
    item = KnowledgeItem(
        content="Pending item",
        content_type=ContentType.CONVERSATION_EXTRACT,
        contexts=["test"],
        verified=False,
    )
    db_session.add(item)
    await db_session.flush()

    resp = await admin_client.get(f"{BASE}/knowledge/pending")
    assert resp.status_code == 200


@pytest.mark.asyncio
async def test_knowledge_management_delete(admin_client, db_session):
    """Delete a knowledge item."""
    item = KnowledgeItem(
        content="To be deleted",
        content_type=ContentType.MANUAL,
        contexts=["test"],
        verified=False,
    )
    db_session.add(item)
    await db_session.flush()

    resp = await admin_client.delete(f"{BASE}/knowledge/{item.id}")
    assert resp.status_code == 204


# ═══════════════════════════════════════════════════════════════════════════════
# 4. Analytics endpoints
# ═══════════════════════════════════════════════════════════════════════════════


@pytest.mark.asyncio
async def test_analytics_overview(admin_client):
    """Overview analytics returns aggregated stats."""
    resp = await admin_client.get(f"{BASE}/analytics/overview")
    assert resp.status_code == 200
    data = resp.json()["data"]
    assert "total_rules" in data
    assert "total_contexts" in data
    assert "total_knowledge_items" in data
    assert "total_conversations" in data


@pytest.mark.asyncio
async def test_analytics_rules(admin_client):
    """Rule performance analytics endpoint."""
    resp = await admin_client.get(f"{BASE}/analytics/rules")
    assert resp.status_code == 200
    assert "data" in resp.json()


@pytest.mark.asyncio
async def test_analytics_conversations(admin_client):
    """Conversation trends analytics endpoint."""
    resp = await admin_client.get(f"{BASE}/analytics/conversations")
    assert resp.status_code == 200
    data = resp.json()["data"]
    assert "period_days" in data
    assert "total" in data


@pytest.mark.asyncio
async def test_analytics_knowledge(admin_client):
    """Knowledge growth analytics endpoint."""
    resp = await admin_client.get(f"{BASE}/analytics/knowledge")
    assert resp.status_code == 200
    data = resp.json()["data"]
    assert "total" in data
    assert "verified" in data
    assert "pending" in data


@pytest.mark.asyncio
async def test_analytics_contexts(admin_client):
    """Context usage analytics endpoint."""
    resp = await admin_client.get(f"{BASE}/analytics/contexts")
    assert resp.status_code == 200
    assert "data" in resp.json()
