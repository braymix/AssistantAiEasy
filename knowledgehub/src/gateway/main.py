"""KnowledgeHub Gateway – FastAPI application entry point."""

from contextlib import asynccontextmanager
from collections.abc import AsyncGenerator

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from src.config import get_settings
from src.config.logging import get_logger, setup_logging
from src.gateway.middleware.request_id import RequestIdMiddleware
from src.gateway.routes import health, knowledge, detection, query
from src.shared.database import dispose_engine, init_db

logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    settings = get_settings()
    setup_logging(settings)
    logger.info(
        "starting_gateway",
        profile=settings.profile.value,
        database=settings.database_url.split("@")[-1] if "@" in settings.database_url else "sqlite",
    )
    await init_db()
    yield
    await dispose_engine()
    logger.info("gateway_shutdown")


def create_app() -> FastAPI:
    settings = get_settings()

    app = FastAPI(
        title=settings.app_name,
        version="0.1.0",
        description="AI-powered knowledge base with context detection",
        lifespan=lifespan,
        docs_url="/docs" if settings.debug else None,
        redoc_url="/redoc" if settings.debug else None,
    )

    # -- Middleware ----------------------------------------------------------
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    app.add_middleware(RequestIdMiddleware)

    # -- Routes -------------------------------------------------------------
    app.include_router(health.router, tags=["health"])
    app.include_router(knowledge.router, prefix="/api/v1/knowledge", tags=["knowledge"])
    app.include_router(detection.router, prefix="/api/v1/detection", tags=["detection"])
    app.include_router(query.router, prefix="/api/v1/query", tags=["query"])

    return app


app = create_app()
