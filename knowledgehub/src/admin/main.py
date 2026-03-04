"""KnowledgeHub Admin – FastAPI application.

Serves both:
  1. HTML dashboard (legacy, unprotected) at ``/`` and ``/rules``
  2. REST API (API-key protected) at ``/api/v1/admin/``
"""

from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import APIRouter, Depends, FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.templating import Jinja2Templates

from src.admin.dependencies import require_api_key
from src.admin.routes import dashboard
from src.admin.routes import analytics as analytics_routes
from src.admin.routes import contexts as contexts_routes
from src.admin.routes import knowledge as knowledge_routes
from src.admin.routes import rules as rules_routes
from src.admin.routes import ui as ui_routes
from src.config import get_settings
from src.config.logging import get_logger, setup_logging
from src.gateway.middleware.logging import LoggingMiddleware
from src.shared.database import dispose_engine, init_db

logger = get_logger(__name__)

TEMPLATES_DIR = Path(__file__).parent / "templates"


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    settings = get_settings()
    setup_logging(settings)
    logger.info("starting_admin", profile=settings.profile.value)
    await init_db()
    yield
    await dispose_engine()


def create_admin_app() -> FastAPI:
    settings = get_settings()

    app = FastAPI(
        title=f"{settings.app_name} Admin",
        version="0.1.0",
        lifespan=lifespan,
        docs_url="/api/docs",
        redoc_url="/api/redoc",
        openapi_url="/api/openapi.json",
    )

    # ── CORS ───────────────────────────────────────────────────────────
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # ── Request logging ────────────────────────────────────────────────
    app.add_middleware(LoggingMiddleware)

    # ── Templates ──────────────────────────────────────────────────────
    templates = Jinja2Templates(directory=str(TEMPLATES_DIR))
    app.state.templates = templates

    # ── HTML dashboard (legacy, no auth) ───────────────────────────────
    app.include_router(dashboard.router, tags=["dashboard"])

    # ── UI routes (templates + HTMX, no auth) ─────────────────────────
    app.include_router(ui_routes.router, tags=["ui"])

    # ── REST API (API-key protected) ───────────────────────────────────
    api_router = APIRouter(
        prefix="/api/v1/admin",
        dependencies=[Depends(require_api_key)],
    )
    api_router.include_router(rules_routes.router, tags=["rules"])
    api_router.include_router(contexts_routes.router, tags=["contexts"])
    api_router.include_router(knowledge_routes.router, tags=["knowledge"])
    api_router.include_router(analytics_routes.router, tags=["analytics"])
    app.include_router(api_router)

    return app


app = create_admin_app()
