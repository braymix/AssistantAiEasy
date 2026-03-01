"""
SQLAlchemy 2.0 async database setup with SQLite / PostgreSQL support.

Engine and session factory are lazily initialised on first use and
can be disposed via ``dispose_engine()`` for clean shutdown.
``AsyncSessionLocal`` is the public session factory alias expected
by the rest of the codebase.
"""

from collections.abc import AsyncGenerator

from sqlalchemy import MetaData, event
from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)
from sqlalchemy.orm import DeclarativeBase

from src.config.settings import get_settings

# Naming convention for constraints (helps Alembic migrations)
convention = {
    "ix": "ix_%(column_0_label)s",
    "uq": "uq_%(table_name)s_%(column_0_name)s",
    "ck": "ck_%(table_name)s_%(constraint_name)s",
    "fk": "fk_%(table_name)s_%(column_0_name)s_%(referred_table_name)s",
    "pk": "pk_%(table_name)s",
}


class Base(DeclarativeBase):
    metadata = MetaData(naming_convention=convention)


# ---------------------------------------------------------------------------
# Engine helpers
# ---------------------------------------------------------------------------

def _build_engine_kwargs() -> dict:
    settings = get_settings()
    kwargs: dict = {
        "echo": settings.db_echo,
    }
    if settings.is_sqlite:
        kwargs["connect_args"] = {"check_same_thread": False}
    else:
        kwargs["pool_size"] = settings.db_pool_size
        kwargs["max_overflow"] = settings.db_max_overflow
    return kwargs


_engine: AsyncEngine | None = None
_session_factory: async_sessionmaker[AsyncSession] | None = None


def _get_engine() -> AsyncEngine:
    global _engine
    if _engine is None:
        settings = get_settings()
        _engine = create_async_engine(settings.database_url, **_build_engine_kwargs())

        # SQLite: enable WAL mode + foreign key enforcement on every connection
        if settings.is_sqlite:
            @event.listens_for(_engine.sync_engine, "connect")
            def _set_sqlite_pragma(dbapi_conn, connection_record):
                cursor = dbapi_conn.cursor()
                cursor.execute("PRAGMA journal_mode=WAL")
                cursor.execute("PRAGMA foreign_keys=ON")
                cursor.close()

    return _engine


def _get_session_factory() -> async_sessionmaker[AsyncSession]:
    global _session_factory
    if _session_factory is None:
        _session_factory = async_sessionmaker(
            bind=_get_engine(),
            class_=AsyncSession,
            expire_on_commit=False,
        )
    return _session_factory


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def AsyncSessionLocal() -> AsyncSession:
    """Return a new ``AsyncSession`` – the canonical session factory.

    Usage::

        async with AsyncSessionLocal() as session:
            ...
    """
    return _get_session_factory()()


async def get_db_session() -> AsyncGenerator[AsyncSession, None]:
    """FastAPI dependency that yields an async database session."""
    async with AsyncSessionLocal() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise


async def init_db() -> None:
    """Create all tables (for initial setup / testing).

    Imports ``src.shared.models`` so that every model is registered on
    ``Base.metadata`` before ``create_all`` runs.
    """
    import src.shared.models  # noqa: F401 – register all models

    engine = _get_engine()
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)


async def dispose_engine() -> None:
    """Dispose of the engine (for graceful shutdown)."""
    global _engine, _session_factory
    if _engine is not None:
        await _engine.dispose()
        _engine = None
        _session_factory = None
