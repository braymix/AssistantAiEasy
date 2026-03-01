"""Context detection endpoints."""

from fastapi import APIRouter, Depends
from sqlalchemy.ext.asyncio import AsyncSession

from src.detection.engine import DetectionEngine
from src.gateway.schemas.detection import DetectionRequest, DetectionResult
from src.shared.database import get_db_session

router = APIRouter()


def _engine(session: AsyncSession = Depends(get_db_session)) -> DetectionEngine:
    return DetectionEngine(session)


@router.post("/detect", response_model=DetectionResult)
async def detect_context(
    payload: DetectionRequest,
    engine: DetectionEngine = Depends(_engine),
) -> DetectionResult:
    """Analyze input text and detect relevant context / triggers."""
    result = await engine.detect(payload.text, payload.context)
    return result
