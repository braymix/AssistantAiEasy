"""Tests for health check endpoints."""

import pytest
from httpx import AsyncClient


@pytest.mark.asyncio
async def test_health_check(client: AsyncClient):
    response = await client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert "profile" in data
    assert "version" in data
    assert "llm_backend" in data
    assert "vectorstore" in data


@pytest.mark.asyncio
async def test_readiness(client: AsyncClient):
    response = await client.get("/health/ready")
    assert response.status_code == 200
    data = response.json()
    # LLM backend won't be running in tests, so status should be "degraded"
    assert data["status"] in ("ready", "degraded")
    assert "checks" in data
