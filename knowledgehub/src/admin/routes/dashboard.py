"""Admin dashboard route."""

from fastapi import APIRouter, Depends, Request
from fastapi.responses import HTMLResponse
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from src.config import get_settings
from src.shared.models import Context, Conversation, DetectionRule, Document, KnowledgeItem
from src.shared.database import get_db_session

router = APIRouter()


@router.get("/", response_class=HTMLResponse)
async def dashboard(
    request: Request,
    session: AsyncSession = Depends(get_db_session),
):
    settings = get_settings()

    doc_count = (await session.execute(select(func.count()).select_from(Document))).scalar_one()
    rule_count = (await session.execute(select(func.count()).select_from(DetectionRule))).scalar_one()
    conv_count = (await session.execute(select(func.count()).select_from(Conversation))).scalar_one()
    ki_count = (await session.execute(select(func.count()).select_from(KnowledgeItem))).scalar_one()
    ctx_count = (await session.execute(select(func.count()).select_from(Context))).scalar_one()

    templates = request.app.state.templates
    return templates.TemplateResponse(
        "dashboard.html",
        {
            "request": request,
            "profile": settings.profile.value,
            "doc_count": doc_count,
            "rule_count": rule_count,
            "conv_count": conv_count,
            "ki_count": ki_count,
            "ctx_count": ctx_count,
            "llm_backend": settings.llm_backend.value,
            "vectorstore": settings.vectorstore_backend.value,
        },
    )
