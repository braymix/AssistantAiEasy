"""
Step 3 — Migrate ChromaDB → Qdrant.

Exports all documents and embeddings from ChromaDB and imports them
into Qdrant with batched upserts:

1. Connect to ChromaDB and list collections
2. Export all documents with embeddings and metadata
3. Create Qdrant collection with correct vector dimension
4. Batch-import points
5. Verify via similarity search

Usage::

    python scripts/migrate_enterprise/03_migrate_vectorstore.py
    python scripts/migrate_enterprise/03_migrate_vectorstore.py --dry-run
    python scripts/migrate_enterprise/03_migrate_vectorstore.py --batch-size 100
"""

from __future__ import annotations

import asyncio
import logging
import os
import sys
import time
from dataclasses import dataclass, field
from typing import Any

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger("migrate.vectorstore")


@dataclass
class VectorDocument:
    """A document extracted from the source vector store."""

    id: str
    embedding: list[float]
    content: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# ChromaDB export
# ---------------------------------------------------------------------------

def export_from_chromadb(
    host: str = "chroma",
    port: int = 8100,
    collection_name: str = "knowledgehub",
    persist_dir: str = "./data/chroma",
) -> list[VectorDocument]:
    """Export all documents from a ChromaDB collection.

    Tries HTTP client first, falls back to persistent local client.
    """
    documents: list[VectorDocument] = []

    try:
        import chromadb

        # Try HTTP client (running ChromaDB server)
        try:
            client = chromadb.HttpClient(host=host, port=port)
            client.heartbeat()
            logger.info("Connected to ChromaDB server at %s:%d", host, port)
        except Exception:
            # Fall back to persistent client (local files)
            logger.info("Using ChromaDB persistent client: %s", persist_dir)
            client = chromadb.PersistentClient(path=persist_dir)

        # List collections
        collections = client.list_collections()
        logger.info("Found %d collection(s)", len(collections))

        # Get target collection
        try:
            collection = client.get_collection(collection_name)
        except Exception:
            logger.warning("Collection '%s' not found — trying first available", collection_name)
            if collections:
                collection = collections[0]
                logger.info("Using collection: %s", collection.name)
            else:
                logger.warning("No collections found in ChromaDB")
                return documents

        # Export all documents
        result = collection.get(include=["embeddings", "documents", "metadatas"])

        ids = result.get("ids", [])
        embeddings = result.get("embeddings", [])
        docs = result.get("documents", [])
        metadatas = result.get("metadatas", [])

        for i, doc_id in enumerate(ids):
            emb = embeddings[i] if embeddings and i < len(embeddings) else []
            content = docs[i] if docs and i < len(docs) else ""
            meta = metadatas[i] if metadatas and i < len(metadatas) else {}

            documents.append(VectorDocument(
                id=doc_id,
                embedding=emb or [],
                content=content or "",
                metadata=meta or {},
            ))

        logger.info(
            "Exported %d documents from ChromaDB (dimension: %d)",
            len(documents),
            len(documents[0].embedding) if documents and documents[0].embedding else 0,
        )

    except ImportError:
        logger.error("chromadb package not installed — run: pip install chromadb")
    except Exception as exc:
        logger.error("ChromaDB export failed: %s", exc)

    return documents


# ---------------------------------------------------------------------------
# Qdrant import
# ---------------------------------------------------------------------------

async def import_to_qdrant(
    documents: list[VectorDocument],
    host: str = "qdrant",
    port: int = 6333,
    collection_name: str = "knowledgehub",
    batch_size: int = 100,
    dry_run: bool = False,
) -> dict[str, Any]:
    """Import documents into a Qdrant collection.

    Creates the collection if it does not exist, then batch-upserts
    all points.
    """
    if not documents:
        logger.warning("No documents to import")
        return {"imported": 0}

    # Determine vector dimension from first document
    dimension = len(documents[0].embedding) if documents[0].embedding else 384
    url = f"http://{host}:{port}"

    if dry_run:
        logger.info(
            "DRY RUN: would import %d documents to Qdrant (%s, dim=%d)",
            len(documents), collection_name, dimension,
        )
        return {"dry_run": True, "would_import": len(documents), "dimension": dimension}

    try:
        import httpx

        async with httpx.AsyncClient(base_url=url, timeout=60) as client:
            # Check if collection exists
            resp = await client.get(f"/collections/{collection_name}")
            collection_exists = resp.status_code == 200

            # Create collection if needed
            if not collection_exists:
                create_payload = {
                    "vectors": {
                        "size": dimension,
                        "distance": "Cosine",
                    },
                }
                resp = await client.put(
                    f"/collections/{collection_name}",
                    json=create_payload,
                )
                if resp.status_code not in (200, 201):
                    raise RuntimeError(f"Failed to create collection: {resp.text}")
                logger.info(
                    "Created Qdrant collection: %s (dim=%d, cosine)",
                    collection_name, dimension,
                )
            else:
                logger.info("Qdrant collection '%s' already exists", collection_name)

            # Batch upsert
            total_imported = 0
            for i in range(0, len(documents), batch_size):
                batch = documents[i : i + batch_size]
                points = []
                for doc in batch:
                    # Clean metadata: Qdrant requires flat values
                    payload: dict[str, Any] = {"content": doc.content}
                    for k, v in doc.metadata.items():
                        if isinstance(v, (str, int, float, bool)):
                            payload[k] = v
                        elif v is not None:
                            payload[k] = str(v)

                    point: dict[str, Any] = {
                        "id": doc.id if _is_uuid(doc.id) else _string_to_uuid_int(doc.id),
                        "vector": doc.embedding,
                        "payload": payload,
                    }
                    points.append(point)

                resp = await client.put(
                    f"/collections/{collection_name}/points",
                    json={"points": points},
                )
                if resp.status_code not in (200, 201):
                    logger.error("Batch upsert failed at %d: %s", i, resp.text)
                    continue

                total_imported += len(batch)
                logger.debug("  Imported batch %d–%d", i, i + len(batch))

            logger.info("Imported %d/%d documents to Qdrant", total_imported, len(documents))
            return {"imported": total_imported, "total": len(documents)}

    except ImportError:
        logger.error("httpx not installed — run: pip install httpx")
        return {"error": "httpx not installed"}
    except Exception as exc:
        logger.error("Qdrant import failed: %s", exc)
        return {"error": str(exc)}


def _is_uuid(s: str) -> bool:
    """Check if a string looks like a UUID."""
    import re
    return bool(re.match(
        r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$",
        s, re.IGNORECASE,
    ))


def _string_to_uuid_int(s: str) -> int:
    """Convert a non-UUID string ID to a positive integer for Qdrant."""
    import hashlib
    return int(hashlib.md5(s.encode()).hexdigest()[:16], 16)  # noqa: S324


# ---------------------------------------------------------------------------
# Verification
# ---------------------------------------------------------------------------

async def verify_migration(
    documents: list[VectorDocument],
    host: str = "qdrant",
    port: int = 6333,
    collection_name: str = "knowledgehub",
    num_tests: int = 3,
) -> bool:
    """Verify migration by running similarity searches.

    Picks a few random documents, searches Qdrant with their
    embeddings, and checks that the original document appears
    in the results.
    """
    if not documents:
        return True

    import random
    url = f"http://{host}:{port}"
    test_docs = random.sample(documents, min(num_tests, len(documents)))

    try:
        import httpx

        async with httpx.AsyncClient(base_url=url, timeout=30) as client:
            # Check collection info
            resp = await client.get(f"/collections/{collection_name}")
            if resp.status_code != 200:
                logger.error("Collection not found: %s", collection_name)
                return False

            info = resp.json().get("result", {})
            point_count = info.get("points_count", 0)
            logger.info(
                "Qdrant collection: %s (%d points)",
                collection_name, point_count,
            )

            if point_count == 0:
                logger.error("Collection is empty!")
                return False

            # Run similarity searches
            passed = 0
            for doc in test_docs:
                if not doc.embedding:
                    continue

                resp = await client.post(
                    f"/collections/{collection_name}/points/search",
                    json={
                        "vector": doc.embedding,
                        "limit": 5,
                        "with_payload": True,
                    },
                )
                if resp.status_code != 200:
                    logger.warning("Search failed for doc %s", doc.id)
                    continue

                results = resp.json().get("result", [])
                # Check if original content appears in top results
                found = any(
                    r.get("payload", {}).get("content", "")[:100] == doc.content[:100]
                    for r in results
                )
                if found:
                    passed += 1
                    logger.info("  [+] Similarity search OK for doc %s", doc.id[:12])
                else:
                    logger.warning("  [!] Doc %s not found in top results", doc.id[:12])

            success = passed >= len(test_docs) * 0.5
            if success:
                logger.info("Verification PASSED (%d/%d tests)", passed, len(test_docs))
            else:
                logger.error("Verification FAILED (%d/%d tests)", passed, len(test_docs))
            return success

    except ImportError:
        logger.error("httpx not installed")
        return False
    except Exception as exc:
        logger.error("Verification failed: %s", exc)
        return False


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

async def migrate_vectorstore(
    dry_run: bool = False,
    batch_size: int = 100,
) -> dict[str, Any]:
    """Execute the ChromaDB → Qdrant migration."""
    logger.info("=" * 60)
    logger.info("KnowledgeHub — VectorStore Migration (ChromaDB → Qdrant)")
    if dry_run:
        logger.info("MODE: DRY RUN")
    logger.info("=" * 60)

    chroma_host = os.environ.get("KNOWLEDGEHUB_CHROMA_HOST", "chroma")
    chroma_port = int(os.environ.get("KNOWLEDGEHUB_CHROMA_PORT", "8100"))
    chroma_collection = os.environ.get("KNOWLEDGEHUB_CHROMA_COLLECTION", "knowledgehub")
    chroma_persist = os.environ.get("KNOWLEDGEHUB_CHROMA_PERSIST_DIR", "./data/chroma")

    qdrant_host = os.environ.get("KNOWLEDGEHUB_QDRANT_HOST", "qdrant")
    qdrant_port = int(os.environ.get("KNOWLEDGEHUB_QDRANT_PORT", "6333"))
    qdrant_collection = os.environ.get("KNOWLEDGEHUB_QDRANT_COLLECTION", "knowledgehub")

    # 1. Export from ChromaDB
    start = time.time()
    documents = export_from_chromadb(
        host=chroma_host,
        port=chroma_port,
        collection_name=chroma_collection,
        persist_dir=chroma_persist,
    )
    export_time = time.time() - start
    logger.info("Export completed in %.1fs (%d documents)", export_time, len(documents))

    if not documents:
        logger.warning("No documents exported — nothing to migrate")
        return {"exported": 0, "imported": 0}

    # 2. Import to Qdrant
    start = time.time()
    import_result = await import_to_qdrant(
        documents=documents,
        host=qdrant_host,
        port=qdrant_port,
        collection_name=qdrant_collection,
        batch_size=batch_size,
        dry_run=dry_run,
    )
    import_time = time.time() - start
    logger.info("Import completed in %.1fs", import_time)

    # 3. Verify
    verified = False
    if not dry_run and import_result.get("imported", 0) > 0:
        verified = await verify_migration(
            documents=documents,
            host=qdrant_host,
            port=qdrant_port,
            collection_name=qdrant_collection,
        )

    logger.info("-" * 60)
    result = {
        "exported": len(documents),
        "import_result": import_result,
        "verified": verified,
        "export_seconds": round(export_time, 2),
        "import_seconds": round(import_time, 2),
    }

    if dry_run:
        logger.info("Dry run complete")
    elif verified:
        logger.info("VectorStore migration SUCCESSFUL")
    else:
        logger.error("VectorStore migration completed with issues")

    return result


def main() -> None:
    dry_run = "--dry-run" in sys.argv
    batch_size = 100
    for i, arg in enumerate(sys.argv):
        if arg == "--batch-size" and i + 1 < len(sys.argv):
            batch_size = int(sys.argv[i + 1])

    result = asyncio.run(migrate_vectorstore(dry_run=dry_run, batch_size=batch_size))
    if not result.get("verified", True) and not dry_run:
        sys.exit(1)


if __name__ == "__main__":
    main()
