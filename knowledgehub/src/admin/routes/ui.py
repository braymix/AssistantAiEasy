"""UI routes – serve Jinja2 templates and handle HTMX partial updates."""

from __future__ import annotations

import json
import uuid

from fastapi import APIRouter, Depends, Form, Query, Request
from fastapi.responses import HTMLResponse, RedirectResponse, Response
from sqlalchemy import String, cast, func, select
from sqlalchemy.ext.asyncio import AsyncSession

from src.config import get_settings
from src.config.logging import get_logger
from src.detection.engine import DetectionEngine, _convert_db_rules
from src.detection.rules import DetectionContext
from src.shared.database import get_db_session
from src.shared.models import (
    Context,
    Conversation,
    DetectionRule,
    Document,
    KnowledgeItem,
    Message,
    RuleType,
)

logger = get_logger(__name__)

router = APIRouter()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _templates(request: Request):
    return request.app.state.templates


async def _pending_count(session: AsyncSession) -> int:
    return (await session.execute(
        select(func.count()).select_from(KnowledgeItem).where(KnowledgeItem.verified.is_(False))
    )).scalar_one()


def _ctx_to_dict(ctx: Context, children: list[dict] | None = None) -> dict:
    return {
        "id": ctx.id,
        "name": ctx.name,
        "description": ctx.description,
        "parent_id": ctx.parent_id,
        "children": children or [],
    }


def _build_tree(contexts: list[Context]) -> list[dict]:
    children_map: dict[str | None, list[Context]] = {}
    for c in contexts:
        children_map.setdefault(c.parent_id, []).append(c)

    def recurse(parent_id: str | None) -> list[dict]:
        return [
            _ctx_to_dict(c, recurse(c.id))
            for c in children_map.get(parent_id, [])
        ]
    return recurse(None)


# ═══════════════════════════════════════════════════════════════════════════
# Dashboard
# ═══════════════════════════════════════════════════════════════════════════


@router.get("/ui", response_class=HTMLResponse)
async def ui_dashboard(
    request: Request,
    session: AsyncSession = Depends(get_db_session),
):
    settings = get_settings()
    total_rules = (await session.execute(select(func.count()).select_from(DetectionRule))).scalar_one()
    active_rules = (await session.execute(select(func.count()).select_from(DetectionRule).where(DetectionRule.enabled.is_(True)))).scalar_one()
    total_ki = (await session.execute(select(func.count()).select_from(KnowledgeItem))).scalar_one()
    verified_ki = (await session.execute(select(func.count()).select_from(KnowledgeItem).where(KnowledgeItem.verified.is_(True)))).scalar_one()
    total_conv = (await session.execute(select(func.count()).select_from(Conversation))).scalar_one()
    total_docs = (await session.execute(select(func.count()).select_from(Document))).scalar_one()
    total_ctx = (await session.execute(select(func.count()).select_from(Context))).scalar_one()

    stats = {
        "total_rules": total_rules,
        "active_rules": active_rules,
        "total_knowledge_items": total_ki,
        "verified_knowledge_items": verified_ki,
        "pending_knowledge_items": total_ki - verified_ki,
        "total_conversations": total_conv,
        "total_documents": total_docs,
        "total_contexts": total_ctx,
    }

    return _templates(request).TemplateResponse("dashboard.html", {
        "request": request,
        "active_page": "dashboard",
        "pending_count": total_ki - verified_ki,
        "stats": stats,
        "profile": settings.profile.value,
        "llm_backend": settings.llm_backend.value,
        "vectorstore": settings.vectorstore_backend.value,
    })


# ═══════════════════════════════════════════════════════════════════════════
# Rules – list
# ═══════════════════════════════════════════════════════════════════════════


@router.get("/ui/rules", response_class=HTMLResponse)
async def ui_rules_list(
    request: Request,
    search: str | None = None,
    rule_type: str | None = None,
    enabled: str | None = None,
    session: AsyncSession = Depends(get_db_session),
):
    q = select(DetectionRule)
    if rule_type:
        q = q.where(DetectionRule.rule_type == RuleType(rule_type))
    if enabled == "true":
        q = q.where(DetectionRule.enabled.is_(True))
    elif enabled == "false":
        q = q.where(DetectionRule.enabled.is_(False))
    if search:
        q = q.where(DetectionRule.name.ilike(f"%{search}%"))
    q = q.order_by(DetectionRule.priority.desc())
    rules = list((await session.execute(q)).scalars().all())
    pending = await _pending_count(session)

    return _templates(request).TemplateResponse("rules/list.html", {
        "request": request,
        "active_page": "rules",
        "pending_count": pending,
        "rules": rules,
        "search": search,
        "filter_type": rule_type,
        "filter_enabled": enabled,
    })


# ═══════════════════════════════════════════════════════════════════════════
# Rules – new / edit
# ═══════════════════════════════════════════════════════════════════════════


@router.get("/ui/rules/new", response_class=HTMLResponse)
async def ui_rule_new(
    request: Request,
    session: AsyncSession = Depends(get_db_session),
):
    pending = await _pending_count(session)
    return _templates(request).TemplateResponse("rules/edit.html", {
        "request": request,
        "active_page": "rules",
        "pending_count": pending,
        "rule": None,
        "config_json": "{}",
    })


@router.get("/ui/rules/{rule_id}", response_class=HTMLResponse)
async def ui_rule_edit(
    rule_id: str,
    request: Request,
    session: AsyncSession = Depends(get_db_session),
):
    rule = await session.get(DetectionRule, rule_id)
    if rule is None:
        return RedirectResponse(url="/ui/rules", status_code=303)
    pending = await _pending_count(session)
    config_json = json.dumps(rule.rule_config or {}, indent=2)
    return _templates(request).TemplateResponse("rules/edit.html", {
        "request": request,
        "active_page": "rules",
        "pending_count": pending,
        "rule": rule,
        "config_json": config_json,
    })


@router.post("/ui/rules/new")
async def ui_rule_create(
    request: Request,
    name: str = Form(...),
    rule_type: str = Form("keyword"),
    description: str = Form(""),
    priority: int = Form(0),
    target_contexts: str = Form(""),
    rule_config: str = Form("{}"),
    enabled: str = Form(""),
    session: AsyncSession = Depends(get_db_session),
):
    contexts_list = [c.strip() for c in target_contexts.split(",") if c.strip()]
    try:
        config = json.loads(rule_config)
    except json.JSONDecodeError:
        config = {}

    rule = DetectionRule(
        id=str(uuid.uuid4()),
        name=name,
        description=description or None,
        rule_type=RuleType(rule_type),
        rule_config=config,
        target_contexts=contexts_list,
        priority=priority,
        enabled=enabled == "true",
    )
    session.add(rule)
    await session.commit()
    logger.info("ui_rule_created", rule_id=rule.id, name=rule.name)
    return RedirectResponse(url="/ui/rules", status_code=303)


@router.post("/ui/rules/{rule_id}")
async def ui_rule_update(
    rule_id: str,
    request: Request,
    name: str = Form(...),
    rule_type: str = Form("keyword"),
    description: str = Form(""),
    priority: int = Form(0),
    target_contexts: str = Form(""),
    rule_config: str = Form("{}"),
    enabled: str = Form(""),
    session: AsyncSession = Depends(get_db_session),
):
    rule = await session.get(DetectionRule, rule_id)
    if rule is None:
        return RedirectResponse(url="/ui/rules", status_code=303)

    contexts_list = [c.strip() for c in target_contexts.split(",") if c.strip()]
    try:
        config = json.loads(rule_config)
    except json.JSONDecodeError:
        config = rule.rule_config or {}

    rule.name = name
    rule.description = description or None
    rule.rule_type = RuleType(rule_type)
    rule.rule_config = config
    rule.target_contexts = contexts_list
    rule.priority = priority
    rule.enabled = enabled == "true"

    await session.commit()
    logger.info("ui_rule_updated", rule_id=rule.id)
    return RedirectResponse(url="/ui/rules", status_code=303)


# ═══════════════════════════════════════════════════════════════════════════
# Contexts – list / create / update
# ═══════════════════════════════════════════════════════════════════════════


@router.get("/ui/contexts", response_class=HTMLResponse)
async def ui_contexts_list(
    request: Request,
    session: AsyncSession = Depends(get_db_session),
):
    rows = list((await session.execute(select(Context).order_by(Context.name))).scalars().all())
    tree = _build_tree(rows)
    pending = await _pending_count(session)

    return _templates(request).TemplateResponse("contexts/list.html", {
        "request": request,
        "active_page": "contexts",
        "pending_count": pending,
        "tree": tree,
        "flat_contexts": rows,
        "total": len(rows),
    })


@router.post("/ui/contexts")
async def ui_context_create(
    name: str = Form(...),
    description: str = Form(""),
    parent_id: str = Form(""),
    session: AsyncSession = Depends(get_db_session),
):
    ctx = Context(
        id=str(uuid.uuid4()),
        name=name,
        description=description or None,
        parent_id=parent_id or None,
    )
    session.add(ctx)
    await session.commit()
    logger.info("ui_context_created", context_id=ctx.id, name=ctx.name)
    return RedirectResponse(url="/ui/contexts", status_code=303)


@router.post("/ui/contexts/{context_id}")
async def ui_context_update(
    context_id: str,
    name: str = Form(...),
    description: str = Form(""),
    parent_id: str = Form(""),
    session: AsyncSession = Depends(get_db_session),
):
    ctx = await session.get(Context, context_id)
    if ctx is None:
        return RedirectResponse(url="/ui/contexts", status_code=303)

    ctx.name = name
    ctx.description = description or None
    ctx.parent_id = parent_id or None
    await session.commit()
    logger.info("ui_context_updated", context_id=ctx.id)
    return RedirectResponse(url="/ui/contexts", status_code=303)


# ═══════════════════════════════════════════════════════════════════════════
# Knowledge – list / review
# ═══════════════════════════════════════════════════════════════════════════


@router.get("/ui/knowledge", response_class=HTMLResponse)
async def ui_knowledge_list(
    request: Request,
    search: str | None = None,
    verified: str | None = None,
    content_type: str | None = None,
    limit: int = 20,
    offset: int = 0,
    session: AsyncSession = Depends(get_db_session),
):
    q = select(KnowledgeItem)
    if verified == "true":
        q = q.where(KnowledgeItem.verified.is_(True))
    elif verified == "false":
        q = q.where(KnowledgeItem.verified.is_(False))
    if content_type:
        from src.shared.models import ContentType as CT
        q = q.where(KnowledgeItem.content_type == CT(content_type))
    if search:
        q = q.where(KnowledgeItem.content.ilike(f"%{search}%"))

    total = (await session.execute(select(func.count()).select_from(q.subquery()))).scalar_one()
    items = list((await session.execute(
        q.order_by(KnowledgeItem.created_at.desc()).offset(offset).limit(limit)
    )).scalars().all())
    pending = await _pending_count(session)

    return _templates(request).TemplateResponse("knowledge/list.html", {
        "request": request,
        "active_page": "knowledge",
        "pending_count": pending,
        "items": items,
        "total": total,
        "limit": limit,
        "offset": offset,
        "search": search,
        "filter_verified": verified,
        "filter_type": content_type,
    })


@router.get("/ui/knowledge/review", response_class=HTMLResponse)
async def ui_knowledge_review(
    request: Request,
    session: AsyncSession = Depends(get_db_session),
):
    items = list((await session.execute(
        select(KnowledgeItem)
        .where(KnowledgeItem.verified.is_(False))
        .order_by(KnowledgeItem.created_at.desc())
        .limit(50)
    )).scalars().all())
    pending = await _pending_count(session)

    return _templates(request).TemplateResponse("knowledge/review.html", {
        "request": request,
        "active_page": "review",
        "pending_count": pending,
        "items": items,
    })


# ═══════════════════════════════════════════════════════════════════════════
# HTMX partials – rule toggle, delete, test
# ═══════════════════════════════════════════════════════════════════════════


@router.post("/ui/htmx/rules/{rule_id}/toggle", response_class=HTMLResponse)
async def htmx_rule_toggle(
    rule_id: str,
    request: Request,
    session: AsyncSession = Depends(get_db_session),
):
    rule = await session.get(DetectionRule, rule_id)
    if rule is None:
        return HTMLResponse("")
    rule.enabled = not rule.enabled
    await session.commit()
    await session.refresh(rule)
    return _templates(request).TemplateResponse("partials/rule_row.html", {
        "request": request,
        "rule": rule,
    })


@router.delete("/ui/htmx/rules/{rule_id}", response_class=HTMLResponse)
async def htmx_rule_delete(
    rule_id: str,
    session: AsyncSession = Depends(get_db_session),
):
    rule = await session.get(DetectionRule, rule_id)
    if rule:
        await session.delete(rule)
        await session.commit()
    return HTMLResponse("", headers={"X-Toast-Message": "Rule deleted", "X-Toast-Type": "success"})


@router.post("/ui/htmx/rules/{rule_id}/test")
async def htmx_rule_test(
    rule_id: str,
    request: Request,
    session: AsyncSession = Depends(get_db_session),
):
    body = await request.json()
    text = body.get("text", "")
    db_rule = await session.get(DetectionRule, rule_id)
    if db_rule is None:
        return {"matched": False, "error": "Rule not found"}

    converted = _convert_db_rules([db_rule])
    if not converted:
        return {"matched": False, "error": "Rule could not be converted"}

    try:
        match = await converted[0].match(text, DetectionContext())
        if match:
            return {
                "matched": True,
                "confidence": match.confidence,
                "extracted": match.extracted,
                "contexts": match.contexts,
            }
        return {"matched": False}
    except Exception as exc:
        return {"matched": False, "error": str(exc)}


@router.post("/ui/htmx/reload-rules", response_class=HTMLResponse)
async def htmx_reload_rules(
    session: AsyncSession = Depends(get_db_session),
):
    engine = DetectionEngine(session)
    await engine.load_rules_from_db()
    count = len(engine._injected_rules)
    return HTMLResponse(
        "",
        headers={"X-Toast-Message": f"Reloaded {count} rules", "X-Toast-Type": "success"},
    )


# ═══════════════════════════════════════════════════════════════════════════
# HTMX partials – context delete
# ═══════════════════════════════════════════════════════════════════════════


@router.delete("/ui/htmx/contexts/{context_id}", response_class=HTMLResponse)
async def htmx_context_delete(
    context_id: str,
    session: AsyncSession = Depends(get_db_session),
):
    ctx = await session.get(Context, context_id)
    if ctx is None:
        return HTMLResponse("", headers={"X-Toast-Message": "Context not found", "X-Toast-Type": "error"})

    # Check children
    child_count = (await session.execute(
        select(func.count()).select_from(Context).where(Context.parent_id == context_id)
    )).scalar_one()
    if child_count > 0:
        return HTMLResponse(
            f'<div id="ctx-{context_id}" class="p-3 rounded-lg bg-red-50 dark:bg-red-900/20 text-red-700 dark:text-red-300 text-sm">Cannot delete: has {child_count} child context(s)</div>',
            headers={"X-Toast-Message": "Cannot delete: has children", "X-Toast-Type": "error"},
        )

    await session.delete(ctx)
    await session.commit()
    return HTMLResponse("", headers={"X-Toast-Message": "Context deleted", "X-Toast-Type": "success"})


# ═══════════════════════════════════════════════════════════════════════════
# HTMX partials – knowledge verify, reject, delete
# ═══════════════════════════════════════════════════════════════════════════


@router.post("/ui/htmx/knowledge/{item_id}/verify", response_class=HTMLResponse)
async def htmx_knowledge_verify(
    item_id: str,
    session: AsyncSession = Depends(get_db_session),
):
    ki = await session.get(KnowledgeItem, item_id)
    if ki is None:
        return HTMLResponse("")
    ki.verified = True
    ki.created_by = ki.created_by or "admin"
    await session.commit()
    return HTMLResponse(
        f'<div id="review-{item_id}" class="p-3 rounded-lg bg-green-50 dark:bg-green-900/20 text-green-700 dark:text-green-300 text-sm flex items-center gap-2">'
        f'<svg class="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M5 13l4 4L19 7"/></svg>'
        f'Approved</div>',
        headers={"X-Toast-Message": "Item approved", "X-Toast-Type": "success"},
    )


@router.post("/ui/htmx/knowledge/{item_id}/reject", response_class=HTMLResponse)
async def htmx_knowledge_reject(
    item_id: str,
    session: AsyncSession = Depends(get_db_session),
):
    ki = await session.get(KnowledgeItem, item_id)
    if ki is None:
        return HTMLResponse("")
    await session.delete(ki)
    await session.commit()
    return HTMLResponse(
        f'<div id="review-{item_id}" class="p-3 rounded-lg bg-red-50 dark:bg-red-900/20 text-red-700 dark:text-red-300 text-sm flex items-center gap-2">'
        f'<svg class="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M6 18L18 6M6 6l12 12"/></svg>'
        f'Rejected</div>',
        headers={"X-Toast-Message": "Item rejected", "X-Toast-Type": "info"},
    )


@router.delete("/ui/htmx/knowledge/{item_id}", response_class=HTMLResponse)
async def htmx_knowledge_delete(
    item_id: str,
    session: AsyncSession = Depends(get_db_session),
):
    ki = await session.get(KnowledgeItem, item_id)
    if ki:
        await session.delete(ki)
        await session.commit()
    return HTMLResponse("", headers={"X-Toast-Message": "Item deleted", "X-Toast-Type": "success"})
