"""Fixtures for the Admin API test suite."""

from __future__ import annotations

import os
from collections.abc import AsyncGenerator
from unittest.mock import patch

import pytest
from httpx import ASGITransport, AsyncClient
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

os.environ.setdefault("KNOWLEDGEHUB_PROFILE", "mini")
os.environ.setdefault("KNOWLEDGEHUB_DATABASE_URL", "sqlite+aiosqlite:///:memory:")

from src.shared.database import Base  # noqa: E402
import src.shared.models  # noqa: E402, F401

TEST_DB_URL = "sqlite+aiosqlite:///:memory:"
TEST_API_KEY = "test-admin-key-12345"


@pytest.fixture
async def db_engine():
    engine = create_async_engine(TEST_DB_URL, echo=False)
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    yield engine
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)
    await engine.dispose()


@pytest.fixture
async def db_session(db_engine) -> AsyncGenerator[AsyncSession, None]:
    factory = async_sessionmaker(bind=db_engine, class_=AsyncSession, expire_on_commit=False)
    async with factory() as session:
        yield session


@pytest.fixture
async def admin_client(db_engine) -> AsyncGenerator[AsyncClient, None]:
    """HTTPX client wired to the admin app with a test DB and API key."""
    factory = async_sessionmaker(bind=db_engine, class_=AsyncSession, expire_on_commit=False)

    async def _override_session():
        async with factory() as session:
            yield session

    # Patch settings to include test API key
    from src.config import get_settings
    settings = get_settings()
    original_keys = settings.api_keys

    settings.api_keys = [TEST_API_KEY]

    from src.admin.main import create_admin_app
    from src.shared.database import get_db_session

    test_app = create_admin_app()
    test_app.dependency_overrides[get_db_session] = _override_session

    transport = ASGITransport(app=test_app)
    async with AsyncClient(
        transport=transport,
        base_url="http://testadmin",
        headers={"X-API-Key": TEST_API_KEY},
    ) as client:
        yield client

    settings.api_keys = original_keys
