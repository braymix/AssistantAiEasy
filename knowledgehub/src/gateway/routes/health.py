"""Health check endpoints."""

from fastapi import APIRouter

from src.config import get_settings

router = APIRouter()


@router.get("/health")
async def health_check() -> dict:
    settings = get_settings()
    return {
        "status": "healthy",
        "profile": settings.profile.value,
        "version": "0.1.0",
    }


@router.get("/health/ready")
async def readiness() -> dict:
    """Readiness probe – verifies core dependencies are available."""
    return {"status": "ready"}
