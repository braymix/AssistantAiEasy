"""Tests for knowledge models."""

import pytest
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from src.knowledge.models import Document


@pytest.mark.asyncio
async def test_create_document(db_session: AsyncSession):
    doc = Document(title="Test Doc", content="Hello world", metadata_json={"source": "test"})
    db_session.add(doc)
    await db_session.flush()

    assert doc.id is not None
    assert len(doc.id) == 36  # UUID format

    result = await db_session.execute(select(Document).where(Document.id == doc.id))
    fetched = result.scalar_one()
    assert fetched.title == "Test Doc"
    assert fetched.content == "Hello world"
    assert fetched.metadata_json == {"source": "test"}
