"""Tests for the VectorStore abstraction.

Uses ChromaVectorStore with a temporary directory so tests are self-contained
and don't depend on any external service.
"""

import os
import tempfile

import pytest

from src.knowledge.vectorstore import (
    ChromaVectorStore,
    DocumentRecord,
    SearchResult,
    VectorStore,
    reset_vector_store,
)


@pytest.fixture
def chroma_dir(tmp_path):
    """Provide a temporary persist directory for Chroma."""
    return str(tmp_path / "chroma_test")


@pytest.fixture
def store(chroma_dir):
    """Create a ChromaVectorStore backed by a temp directory."""
    reset_vector_store()
    return ChromaVectorStore(
        persist_dir=chroma_dir,
        collection_name="test_collection",
    )


# ── ABC interface ──────────────────────────────────────────────────────────


def test_vectorstore_is_abstract():
    """Cannot instantiate VectorStore directly."""
    with pytest.raises(TypeError):
        VectorStore()


# ── add + get ──────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_add_and_get(store):
    ids = await store.add(
        texts=["Hello world", "Goodbye world"],
        metadatas=[{"source": "test"}, {"source": "test"}],
        ids=["id-1", "id-2"],
    )
    assert ids == ["id-1", "id-2"]

    docs = await store.get(["id-1", "id-2"])
    assert len(docs) == 2
    assert isinstance(docs[0], DocumentRecord)
    assert docs[0].content == "Hello world"
    assert docs[1].content == "Goodbye world"


@pytest.mark.asyncio
async def test_add_with_embeddings(store):
    """Pre-computed embeddings should be accepted without error."""
    fake_embeddings = [[0.1] * 384, [0.2] * 384]
    ids = await store.add(
        texts=["text a", "text b"],
        metadatas=[{}, {}],
        ids=["emb-1", "emb-2"],
        embeddings=fake_embeddings,
    )
    assert ids == ["emb-1", "emb-2"]

    docs = await store.get(["emb-1"])
    assert docs[0].content == "text a"


# ── search ─────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_search_returns_results(store):
    embeddings = [[1.0, 0.0, 0.0] + [0.0] * 381, [0.0, 1.0, 0.0] + [0.0] * 381]
    await store.add(
        texts=["Document about cats", "Document about dogs"],
        metadatas=[{"topic": "cats"}, {"topic": "dogs"}],
        ids=["cat-1", "dog-1"],
        embeddings=embeddings,
    )

    query_embedding = [1.0, 0.0, 0.0] + [0.0] * 381
    results = await store.search(query_embedding, n_results=2)

    assert len(results) >= 1
    assert isinstance(results[0], SearchResult)
    assert results[0].content == "Document about cats"
    assert results[0].score > 0


@pytest.mark.asyncio
async def test_search_with_filter(store):
    embeddings = [[1.0] + [0.0] * 383, [1.0] + [0.0] * 383]
    await store.add(
        texts=["Cat text", "Dog text"],
        metadatas=[{"topic": "cats"}, {"topic": "dogs"}],
        ids=["f-cat", "f-dog"],
        embeddings=embeddings,
    )

    query = [1.0] + [0.0] * 383
    results = await store.search(query, n_results=10, filter={"topic": "dogs"})

    assert len(results) == 1
    assert results[0].content == "Dog text"


@pytest.mark.asyncio
async def test_search_empty_collection(store):
    query = [0.5] * 384
    results = await store.search(query, n_results=5)
    assert results == []


# ── delete ─────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_delete(store):
    await store.add(
        texts=["to delete"],
        metadatas=[{}],
        ids=["del-1"],
    )

    ok = await store.delete(["del-1"])
    assert ok is True

    docs = await store.get(["del-1"])
    assert docs == []


# ── update ─────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_update_text(store):
    await store.add(
        texts=["original"],
        metadatas=[{"v": "1"}],
        ids=["upd-1"],
    )

    ok = await store.update("upd-1", text="updated")
    assert ok is True

    docs = await store.get(["upd-1"])
    assert docs[0].content == "updated"


@pytest.mark.asyncio
async def test_update_metadata(store):
    await store.add(
        texts=["doc"],
        metadatas=[{"color": "red"}],
        ids=["upd-meta"],
    )

    ok = await store.update("upd-meta", metadata={"color": "blue"})
    assert ok is True

    docs = await store.get(["upd-meta"])
    assert docs[0].metadata.get("color") == "blue"


# ── get_collection ─────────────────────────────────────────────────────────


def test_get_collection_returns_object(store):
    coll = store.get_collection("another_context")
    assert coll is not None
    assert coll.name == "another_context"
