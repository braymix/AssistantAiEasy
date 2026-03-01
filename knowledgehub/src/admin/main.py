"""KnowledgeHub Admin Dashboard – FastAPI application."""

from contextlib import asynccontextmanager
from collections.abc import AsyncGenerator
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates

from src.config import get_settings
from src.config.logging import get_logger, setup_logging
from src.admin.routes import dashboard, rules
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
    )

    templates = Jinja2Templates(directory=str(TEMPLATES_DIR))
    app.state.templates = templates

    app.include_router(dashboard.router, tags=["dashboard"])
    app.include_router(rules.router, prefix="/rules", tags=["rules"])

    return app


app = create_admin_app()
