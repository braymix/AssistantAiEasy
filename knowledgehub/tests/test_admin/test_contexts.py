"""Tests for the Admin Contexts REST API."""

from __future__ import annotations

import pytest


BASE = "/api/v1/admin"


# ═══════════════════════════════════════════════════════════════════════════
# LIST
# ═══════════════════════════════════════════════════════════════════════════


@pytest.mark.asyncio
async def test_list_contexts_empty(admin_client):
    resp = await admin_client.get(f"{BASE}/contexts")
    assert resp.status_code == 200
    body = resp.json()
    assert body["data"] == []
    assert body["meta"]["total"] == 0


# ═══════════════════════════════════════════════════════════════════════════
# CREATE
# ═══════════════════════════════════════════════════════════════════════════


@pytest.mark.asyncio
async def test_create_context(admin_client):
    resp = await admin_client.post(f"{BASE}/contexts", json={
        "name": "technology",
        "description": "Tech topics",
        "metadata": {"color": "blue"},
    })
    assert resp.status_code == 201
    data = resp.json()["data"]
    assert data["name"] == "technology"
    assert data["description"] == "Tech topics"
    assert data["metadata"]["color"] == "blue"
    assert data["parent_id"] is None


@pytest.mark.asyncio
async def test_create_child_context(admin_client):
    parent_resp = await admin_client.post(f"{BASE}/contexts", json={
        "name": "parent_ctx",
    })
    parent_id = parent_resp.json()["data"]["id"]

    child_resp = await admin_client.post(f"{BASE}/contexts", json={
        "name": "child_ctx",
        "parent_id": parent_id,
    })
    assert child_resp.status_code == 201
    assert child_resp.json()["data"]["parent_id"] == parent_id


@pytest.mark.asyncio
async def test_create_context_invalid_parent(admin_client):
    resp = await admin_client.post(f"{BASE}/contexts", json={
        "name": "orphan",
        "parent_id": "nonexistent",
    })
    assert resp.status_code == 404


# ═══════════════════════════════════════════════════════════════════════════
# UPDATE
# ═══════════════════════════════════════════════════════════════════════════


@pytest.mark.asyncio
async def test_update_context(admin_client):
    create_resp = await admin_client.post(f"{BASE}/contexts", json={
        "name": "updatable",
    })
    ctx_id = create_resp.json()["data"]["id"]

    resp = await admin_client.put(f"{BASE}/contexts/{ctx_id}", json={
        "name": "updated_name",
        "description": "Updated desc",
    })
    assert resp.status_code == 200
    data = resp.json()["data"]
    assert data["name"] == "updated_name"
    assert data["description"] == "Updated desc"


@pytest.mark.asyncio
async def test_update_context_not_found(admin_client):
    resp = await admin_client.put(f"{BASE}/contexts/missing", json={"name": "x"})
    assert resp.status_code == 404


# ═══════════════════════════════════════════════════════════════════════════
# DELETE
# ═══════════════════════════════════════════════════════════════════════════


@pytest.mark.asyncio
async def test_delete_context(admin_client):
    create_resp = await admin_client.post(f"{BASE}/contexts", json={
        "name": "deletable",
    })
    ctx_id = create_resp.json()["data"]["id"]

    del_resp = await admin_client.delete(f"{BASE}/contexts/{ctx_id}")
    assert del_resp.status_code == 204


@pytest.mark.asyncio
async def test_delete_context_not_found(admin_client):
    resp = await admin_client.delete(f"{BASE}/contexts/missing")
    assert resp.status_code == 404


@pytest.mark.asyncio
async def test_delete_context_with_children_blocked(admin_client):
    parent_resp = await admin_client.post(f"{BASE}/contexts", json={
        "name": "parent_del",
    })
    parent_id = parent_resp.json()["data"]["id"]

    await admin_client.post(f"{BASE}/contexts", json={
        "name": "child_del",
        "parent_id": parent_id,
    })

    resp = await admin_client.delete(f"{BASE}/contexts/{parent_id}")
    assert resp.status_code == 409
    assert "child" in resp.json()["detail"].lower()


# ═══════════════════════════════════════════════════════════════════════════
# HIERARCHY
# ═══════════════════════════════════════════════════════════════════════════


@pytest.mark.asyncio
async def test_list_contexts_flat(admin_client):
    await admin_client.post(f"{BASE}/contexts", json={"name": "flat_a"})
    await admin_client.post(f"{BASE}/contexts", json={"name": "flat_b"})

    resp = await admin_client.get(f"{BASE}/contexts", params={"flat": True})
    assert resp.status_code == 200
    names = [c["name"] for c in resp.json()["data"]]
    assert "flat_a" in names
    assert "flat_b" in names
    # Flat mode: children should be empty
    assert all(c["children"] == [] for c in resp.json()["data"])


@pytest.mark.asyncio
async def test_list_contexts_hierarchical(admin_client):
    parent_resp = await admin_client.post(f"{BASE}/contexts", json={"name": "hier_parent"})
    parent_id = parent_resp.json()["data"]["id"]
    await admin_client.post(f"{BASE}/contexts", json={
        "name": "hier_child",
        "parent_id": parent_id,
    })

    resp = await admin_client.get(f"{BASE}/contexts", params={"flat": False})
    assert resp.status_code == 200
    # In hierarchical mode, root nodes appear at top level
    root_names = [c["name"] for c in resp.json()["data"]]
    assert "hier_parent" in root_names
    # hier_child should NOT be at root level
    assert "hier_child" not in root_names


# ═══════════════════════════════════════════════════════════════════════════
# STATS
# ═══════════════════════════════════════════════════════════════════════════


@pytest.mark.asyncio
async def test_context_stats(admin_client):
    create_resp = await admin_client.post(f"{BASE}/contexts", json={
        "name": "stats_ctx",
    })
    ctx_id = create_resp.json()["data"]["id"]

    resp = await admin_client.get(f"{BASE}/contexts/{ctx_id}/stats")
    assert resp.status_code == 200
    data = resp.json()["data"]
    assert data["context_name"] == "stats_ctx"
    assert data["knowledge_count"] == 0
    assert data["rule_count"] == 0
    assert data["conversation_count"] == 0


@pytest.mark.asyncio
async def test_context_stats_not_found(admin_client):
    resp = await admin_client.get(f"{BASE}/contexts/missing/stats")
    assert resp.status_code == 404


# ═══════════════════════════════════════════════════════════════════════════
# KNOWLEDGE for context
# ═══════════════════════════════════════════════════════════════════════════


@pytest.mark.asyncio
async def test_context_knowledge_empty(admin_client):
    create_resp = await admin_client.post(f"{BASE}/contexts", json={
        "name": "know_ctx",
    })
    ctx_id = create_resp.json()["data"]["id"]

    resp = await admin_client.get(f"{BASE}/contexts/{ctx_id}/knowledge")
    assert resp.status_code == 200
    assert resp.json()["data"] == []
    assert resp.json()["meta"]["total"] == 0


@pytest.mark.asyncio
async def test_context_knowledge_not_found(admin_client):
    resp = await admin_client.get(f"{BASE}/contexts/missing/knowledge")
    assert resp.status_code == 404
