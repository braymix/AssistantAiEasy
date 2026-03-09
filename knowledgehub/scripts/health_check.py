"""Health check script for KnowledgeHub services.

Verifies connectivity to all configured services and reports status.
Exit codes:
  0 = all services healthy
  1 = one or more services unhealthy

Usage:
    python scripts/health_check.py            # check all services
    python scripts/health_check.py --json     # JSON output (for monitoring)
    python scripts/health_check.py --verbose  # detailed output
"""

import asyncio
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


async def check_gateway(base_url: str) -> dict:
    """Check Gateway API health."""
    import httpx

    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            start = time.monotonic()
            resp = await client.get(f"{base_url}/health")
            latency_ms = (time.monotonic() - start) * 1000
            if resp.status_code == 200:
                data = resp.json()
                return {
                    "service": "gateway",
                    "healthy": True,
                    "latency_ms": round(latency_ms, 1),
                    "details": data,
                }
            return {
                "service": "gateway",
                "healthy": False,
                "error": f"HTTP {resp.status_code}",
            }
    except Exception as e:
        return {"service": "gateway", "healthy": False, "error": str(e)}


async def check_admin(base_url: str) -> dict:
    """Check Admin API health."""
    import httpx

    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            start = time.monotonic()
            resp = await client.get(f"{base_url}/health")
            latency_ms = (time.monotonic() - start) * 1000
            return {
                "service": "admin",
                "healthy": resp.status_code == 200,
                "latency_ms": round(latency_ms, 1),
            }
    except Exception as e:
        return {"service": "admin", "healthy": False, "error": str(e)}


async def check_ollama(base_url: str) -> dict:
    """Check Ollama API availability."""
    import httpx

    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            start = time.monotonic()
            resp = await client.get(f"{base_url}/api/tags")
            latency_ms = (time.monotonic() - start) * 1000
            if resp.status_code == 200:
                models = resp.json().get("models", [])
                return {
                    "service": "ollama",
                    "healthy": True,
                    "latency_ms": round(latency_ms, 1),
                    "details": {"models_loaded": len(models)},
                }
            return {
                "service": "ollama",
                "healthy": False,
                "error": f"HTTP {resp.status_code}",
            }
    except Exception as e:
        return {"service": "ollama", "healthy": False, "error": str(e)}


async def check_database(database_url: str) -> dict:
    """Check database connectivity."""
    try:
        if "sqlite" in database_url:
            db_path = Path(database_url.split("///")[-1])
            if db_path.exists():
                size_mb = db_path.stat().st_size / (1024 * 1024)
                return {
                    "service": "database",
                    "healthy": True,
                    "details": {"type": "sqlite", "size_mb": round(size_mb, 2)},
                }
            return {
                "service": "database",
                "healthy": False,
                "error": f"SQLite file not found: {db_path}",
            }
        else:
            # PostgreSQL: try a simple connection
            from sqlalchemy.ext.asyncio import create_async_engine
            from sqlalchemy import text

            engine = create_async_engine(database_url)
            start = time.monotonic()
            async with engine.connect() as conn:
                await conn.execute(text("SELECT 1"))
            latency_ms = (time.monotonic() - start) * 1000
            await engine.dispose()
            return {
                "service": "database",
                "healthy": True,
                "latency_ms": round(latency_ms, 1),
                "details": {"type": "postgresql"},
            }
    except Exception as e:
        return {"service": "database", "healthy": False, "error": str(e)}


async def check_chroma(host: str, port: int) -> dict:
    """Check ChromaDB availability."""
    import httpx

    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            start = time.monotonic()
            resp = await client.get(f"http://{host}:{port}/api/v1/heartbeat")
            latency_ms = (time.monotonic() - start) * 1000
            return {
                "service": "chroma",
                "healthy": resp.status_code == 200,
                "latency_ms": round(latency_ms, 1),
            }
    except Exception as e:
        return {"service": "chroma", "healthy": False, "error": str(e)}


async def check_qdrant(host: str, port: int) -> dict:
    """Check Qdrant availability."""
    import httpx

    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            start = time.monotonic()
            resp = await client.get(f"http://{host}:{port}/healthz")
            latency_ms = (time.monotonic() - start) * 1000
            return {
                "service": "qdrant",
                "healthy": resp.status_code == 200,
                "latency_ms": round(latency_ms, 1),
            }
    except Exception as e:
        return {"service": "qdrant", "healthy": False, "error": str(e)}


async def check_openwebui(url: str) -> dict:
    """Check Open WebUI availability."""
    import httpx

    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            start = time.monotonic()
            resp = await client.get(f"{url}/health")
            latency_ms = (time.monotonic() - start) * 1000
            return {
                "service": "open-webui",
                "healthy": resp.status_code == 200,
                "latency_ms": round(latency_ms, 1),
            }
    except Exception as e:
        return {"service": "open-webui", "healthy": False, "error": str(e)}


async def main() -> None:
    from src.config import get_settings
    from src.config.logging import setup_logging

    settings = get_settings()
    setup_logging(settings)

    json_output = "--json" in sys.argv
    verbose = "--verbose" in sys.argv

    # ── Build list of checks based on profile ───────────────────────────────
    checks = []

    # Always check gateway, admin, database
    gateway_url = f"http://{settings.gateway_host}:{settings.gateway_port}"
    admin_url = f"http://{settings.gateway_host}:{settings.admin_port}"

    checks.append(check_gateway(gateway_url))
    checks.append(check_admin(admin_url))
    checks.append(check_database(settings.database_url))

    # LLM backend
    if settings.llm_backend.value == "ollama":
        checks.append(check_ollama(settings.ollama_base_url))

    # Vector store
    if settings.vectorstore_backend.value == "chroma":
        checks.append(check_chroma(settings.chroma_host, settings.chroma_port))
    elif settings.vectorstore_backend.value == "qdrant":
        checks.append(check_qdrant(settings.qdrant_host, settings.qdrant_port))

    # Open WebUI (optional)
    if settings.openwebui_url:
        checks.append(check_openwebui(settings.openwebui_url))

    # ── Run all checks concurrently ─────────────────────────────────────────
    results = await asyncio.gather(*checks, return_exceptions=True)

    # Handle any exceptions from gather
    final_results = []
    for r in results:
        if isinstance(r, Exception):
            final_results.append({"service": "unknown", "healthy": False, "error": str(r)})
        else:
            final_results.append(r)

    # ── Output results ──────────────────────────────────────────────────────
    all_healthy = all(r["healthy"] for r in final_results)

    if json_output:
        import json

        output = {
            "status": "healthy" if all_healthy else "unhealthy",
            "profile": settings.profile.value,
            "services": final_results,
        }
        print(json.dumps(output, indent=2))
    else:
        status_icon = {True: "+", False: "!"}
        print(f"\nKnowledgeHub Health Check (profile: {settings.profile.value})")
        print("=" * 50)

        for r in final_results:
            icon = status_icon[r["healthy"]]
            status = "HEALTHY" if r["healthy"] else "UNHEALTHY"
            latency = f" ({r['latency_ms']}ms)" if "latency_ms" in r else ""
            print(f"  [{icon}] {r['service']:<15} {status}{latency}")

            if verbose and "details" in r:
                for k, v in r["details"].items():
                    print(f"      {k}: {v}")

            if not r["healthy"] and "error" in r:
                print(f"      error: {r['error']}")

        print("=" * 50)
        overall = "ALL HEALTHY" if all_healthy else "DEGRADED"
        print(f"  Overall: {overall}")
        print()

    sys.exit(0 if all_healthy else 1)


if __name__ == "__main__":
    asyncio.run(main())
