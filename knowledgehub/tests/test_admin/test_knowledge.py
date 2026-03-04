"""Tests for the Admin Knowledge REST API."""

from __future__ import annotations

import uuid

import pytest
from sqlalchemy.ext.asyncio import AsyncSession

BASE = "/api/v1/admin"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


async def _seed_knowledge_item(admin_client, db_engine, **overrides):
    """Insert a KnowledgeItem directly via the DB for testing."""
    from sqlalchemy.ext.asyncio import async_sessionmaker, AsyncSession as AS
    from src.shared.models import KnowledgeItem, ContentType

    defaults = {
        "id": str(uuid.uuid4()),
        "content": "Test knowledge content",
        "content_type": ContentType.MANUAL,
        "contexts": ["tech"],
        "verified": False,
        "created_by": "test",
    }
    defaults.update(overrides)

    factory = async_sessionmaker(bind=db_engine, class_=AS, expire_on_commit=False)
    async with factory() as session:
        ki = KnowledgeItem(**defaults)
        session.add(ki)
        await session.commit()
        await session.refresh(ki)
        return ki


# ═══════════════════════════════════════════════════════════════════════════
# LIST
# ═══════════════════════════════════════════════════════════════════════════


@pytest.mark.asyncio
async def test_list_knowledge_empty(admin_client):
    resp = await admin_client.get(f"{BASE}/knowledge")
    assert resp.status_code == 200
    assert resp.json()["data"] == []
    assert resp.json()["meta"]["total"] == 0


@pytest.mark.asyncio
async def test_list_knowledge_with_items(admin_client, db_engine):
    await _seed_knowledge_item(admin_client, db_engine, content="Item A")
    await _seed_knowledge_item(admin_client, db_engine, content="Item B", verified=True)

    resp = await admin_client.get(f"{BASE}/knowledge")
    assert resp.status_code == 200
    assert resp.json()["meta"]["total"] == 2


@pytest.mark.asyncio
async def test_list_knowledge_filter_verified(admin_client, db_engine):
    await _seed_knowledge_item(admin_client, db_engine, content="Unverified")
    await _seed_knowledge_item(admin_client, db_engine, content="Verified", verified=True)

    resp = await admin_client.get(f"{BASE}/knowledge", params={"verified": True})
    assert resp.status_code == 200
    assert all(item["verified"] for item in resp.json()["data"])


@pytest.mark.asyncio
async def test_list_knowledge_search(admin_client, db_engine):
    await _seed_knowledge_item(admin_client, db_engine, content="Python programming guide")
    await _seed_knowledge_item(admin_client, db_engine, content="Cooking recipes")

    resp = await admin_client.get(f"{BASE}/knowledge", params={"search": "Python"})
    assert resp.status_code == 200
    assert resp.json()["meta"]["total"] == 1
    assert "Python" in resp.json()["data"][0]["content"]


# ═══════════════════════════════════════════════════════════════════════════
# PENDING
# ═══════════════════════════════════════════════════════════════════════════


@pytest.mark.asyncio
async def test_list_pending(admin_client, db_engine):
    await _seed_knowledge_item(admin_client, db_engine, content="Pending item", verified=False)
    await _seed_knowledge_item(admin_client, db_engine, content="Approved item", verified=True)

    resp = await admin_client.get(f"{BASE}/knowledge/pending")
    assert resp.status_code == 200
    assert resp.json()["meta"]["total"] == 1
    assert resp.json()["data"][0]["content"] == "Pending item"


# ═══════════════════════════════════════════════════════════════════════════
# DELETE
# ═══════════════════════════════════════════════════════════════════════════


@pytest.mark.asyncio
async def test_delete_knowledge_item(admin_client, db_engine):
    ki = await _seed_knowledge_item(admin_client, db_engine, content="To delete")

    resp = await admin_client.delete(f"{BASE}/knowledge/{ki.id}")
    assert resp.status_code == 204

    # Verify it's gone
    list_resp = await admin_client.get(f"{BASE}/knowledge", params={"search": "To delete"})
    assert list_resp.json()["meta"]["total"] == 0


@pytest.mark.asyncio
async def test_delete_knowledge_not_found(admin_client):
    resp = await admin_client.delete(f"{BASE}/knowledge/nonexistent")
    assert resp.status_code == 404


# ═══════════════════════════════════════════════════════════════════════════
# EXPORT
# ═══════════════════════════════════════════════════════════════════════════


@pytest.mark.asyncio
async def test_export_knowledge(admin_client, db_engine):
    await _seed_knowledge_item(admin_client, db_engine, content="Export A", verified=True)
    await _seed_knowledge_item(admin_client, db_engine, content="Export B", verified=False)

    resp = await admin_client.get(f"{BASE}/knowledge/export")
    assert resp.status_code == 200
    data = resp.json()["data"]
    assert data["total"] == 2
    assert len(data["items"]) == 2
    assert "exported_at" in data


@pytest.mark.asyncio
async def test_export_knowledge_verified_only(admin_client, db_engine):
    await _seed_knowledge_item(admin_client, db_engine, content="Verified export", verified=True)
    await _seed_knowledge_item(admin_client, db_engine, content="Unverified export", verified=False)

    resp = await admin_client.get(f"{BASE}/knowledge/export", params={"verified_only": True})
    assert resp.status_code == 200
    data = resp.json()["data"]
    assert data["total"] == 1
    assert data["items"][0]["verified"] is True


# ═══════════════════════════════════════════════════════════════════════════
# ANALYTICS overview (bonus: quick smoke test)
# ═══════════════════════════════════════════════════════════════════════════


@pytest.mark.asyncio
async def test_analytics_overview(admin_client):
    resp = await admin_client.get(f"{BASE}/analytics/overview")
    assert resp.status_code == 200
    data = resp.json()["data"]
    assert "total_rules" in data
    assert "total_knowledge_items" in data
    assert "total_conversations" in data


@pytest.mark.asyncio
async def test_analytics_rules(admin_client):
    resp = await admin_client.get(f"{BASE}/analytics/rules")
    assert resp.status_code == 200
    assert isinstance(resp.json()["data"], list)


@pytest.mark.asyncio
async def test_analytics_conversations(admin_client):
    resp = await admin_client.get(f"{BASE}/analytics/conversations")
    assert resp.status_code == 200
    data = resp.json()["data"]
    assert "period_days" in data
    assert data["period_days"] == 30


@pytest.mark.asyncio
async def test_analytics_knowledge_growth(admin_client):
    resp = await admin_client.get(f"{BASE}/analytics/knowledge")
    assert resp.status_code == 200
    data = resp.json()["data"]
    assert "total" in data
    assert "verified" in data
    assert "pending" in data


@pytest.mark.asyncio
async def test_analytics_contexts(admin_client):
    resp = await admin_client.get(f"{BASE}/analytics/contexts")
    assert resp.status_code == 200
    assert isinstance(resp.json()["data"], list)
