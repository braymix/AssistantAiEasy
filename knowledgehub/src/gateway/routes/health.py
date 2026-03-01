"""Health check endpoints."""

from fastapi import APIRouter

from src.config import get_settings
from src.llm.base import get_llm_provider

router = APIRouter()


@router.get("/health")
async def health_check() -> dict:
    settings = get_settings()
    return {
        "status": "healthy",
        "profile": settings.profile.value,
        "llm_backend": settings.llm_backend.value,
        "vectorstore": settings.vectorstore_backend.value,
        "version": "0.1.0",
    }


@router.get("/health/ready")
async def readiness() -> dict:
    """Readiness probe – verifies core dependencies are reachable."""
    checks: dict = {}

    # LLM backend
    try:
        llm = get_llm_provider()
        checks["llm"] = await llm.health_check()
    except Exception:
        checks["llm"] = False

    all_ok = all(checks.values())
    return {"status": "ready" if all_ok else "degraded", "checks": checks}
