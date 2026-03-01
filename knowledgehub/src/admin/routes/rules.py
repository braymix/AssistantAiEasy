"""Admin routes for managing detection rules."""

from fastapi import APIRouter, Depends, Request
from fastapi.responses import HTMLResponse, RedirectResponse
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from src.shared.models import DetectionRule
from src.shared.database import get_db_session

router = APIRouter()


@router.get("/", response_class=HTMLResponse)
async def list_rules(
    request: Request,
    session: AsyncSession = Depends(get_db_session),
):
    result = await session.execute(
        select(DetectionRule).order_by(DetectionRule.priority.desc())
    )
    rules = list(result.scalars().all())

    templates = request.app.state.templates
    return templates.TemplateResponse(
        "rules.html",
        {"request": request, "rules": rules},
    )


@router.post("/{rule_id}/toggle")
async def toggle_rule(
    rule_id: str,
    session: AsyncSession = Depends(get_db_session),
):
    rule = await session.get(DetectionRule, rule_id)
    if rule:
        rule.enabled = not rule.enabled
        await session.commit()
    return RedirectResponse(url="/rules", status_code=303)
