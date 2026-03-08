"""
KnowledgeHub Gateway – FastAPI application entry point.

This is the main service that sits between Open WebUI and the LLM backend.
It provides OpenAI-compatible endpoints, context detection, and RAG enrichment.

Startup sequence:
  1. Configure structured logging
  2. Initialise database (create tables if needed)
  3. Verify LLM backend connectivity
  4. Load detection rules into memory
"""

from contextlib import asynccontextmanager
from collections.abc import AsyncGenerator

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from src.config import get_settings
from src.config.logging import get_logger, setup_logging
from src.gateway.middleware.logging import LoggingMiddleware
from src.gateway.routes import chat, detection, health, knowledge, query
from src.llm.base import get_llm_provider
from src.shared.database import dispose_engine, init_db

logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    settings = get_settings()
    setup_logging(settings)

    db_label = settings.database_url.split("@")[-1] if "@" in settings.database_url else "sqlite"
    logger.info(
        "gateway_starting",
        profile=settings.profile.value,
        llm_backend=settings.llm_backend.value,
        vectorstore=settings.vectorstore_backend.value,
        database=db_label,
    )

    # 1. Database
    await init_db()
    logger.info("database_ready")

    # 2. LLM health check (non-blocking – log warning if unreachable)
    llm = get_llm_provider()
    try:
        status = await llm.health_check()
        if status.healthy:
            logger.info("llm_backend_healthy", backend=settings.llm_backend.value)
        else:
            logger.warning("llm_backend_unreachable", backend=settings.llm_backend.value, detail=status.detail)
    except Exception as exc:
        logger.warning("llm_health_check_failed", error=str(exc))

    logger.info("gateway_ready")
    yield

    # Shutdown
    await dispose_engine()
    logger.info("gateway_shutdown")


def create_app() -> FastAPI:
    settings = get_settings()

    app = FastAPI(
        title=settings.app_name,
        version="0.1.0",
        description="AI-powered knowledge base with context detection and LLM proxy",
        lifespan=lifespan,
        docs_url="/docs" if settings.debug else "/docs",
        redoc_url="/redoc" if settings.debug else None,
    )

    # -- Middleware (order matters: outermost first) -------------------------
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    app.add_middleware(LoggingMiddleware)

    # -- Routes -------------------------------------------------------------
    # OpenAI-compatible endpoints (top-level, no prefix)
    app.include_router(chat.router, tags=["chat"])

    # Health
    app.include_router(health.router, tags=["health"])

    # Internal API
    app.include_router(knowledge.router, prefix="/api/v1/knowledge", tags=["knowledge"])
    app.include_router(detection.router, prefix="/api/v1/detection", tags=["detection"])
    app.include_router(query.router, prefix="/api/v1/query", tags=["query"])

    return app


app = create_app()
