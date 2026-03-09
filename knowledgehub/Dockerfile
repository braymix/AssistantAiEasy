# ============================================================================
# KnowledgeHub – Multi-stage Dockerfile
# ============================================================================
# Stages: deps → build → production / dev
# Target image < 500 MB, non-root, health check included
# ============================================================================

# ---------------------------------------------------------------------------
# Stage 1: dependency installer
# ---------------------------------------------------------------------------
FROM python:3.11-slim AS deps

WORKDIR /app

# System deps needed for building wheels
RUN apt-get update && \
    apt-get install -y --no-install-recommends gcc libpq-dev && \
    rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir poetry==1.8.4 && \
    poetry config virtualenvs.in-project true

COPY pyproject.toml poetry.lock* ./
RUN poetry install --no-interaction --no-ansi --only main --no-root

# ---------------------------------------------------------------------------
# Stage 2: production
# ---------------------------------------------------------------------------
FROM python:3.11-slim AS production

LABEL maintainer="KnowledgeHub Team"
LABEL description="KnowledgeHub Gateway + Admin API"

WORKDIR /app

# Runtime deps only
RUN apt-get update && \
    apt-get install -y --no-install-recommends libpq5 curl && \
    rm -rf /var/lib/apt/lists/*

# Copy virtualenv from deps stage
COPY --from=deps /app/.venv /app/.venv
ENV PATH="/app/.venv/bin:$PATH"
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Copy application code
COPY src/ ./src/
COPY scripts/ ./scripts/
COPY alembic.ini* ./

# Create non-root user
RUN groupadd -r khub && useradd -r -g khub -d /app -s /sbin/nologin khub && \
    mkdir -p /app/data/sqlite /app/data/logs /app/data/chroma && \
    chown -R khub:khub /app

USER khub

# Health check — gateway endpoint
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD ["curl", "-sf", "http://localhost:8000/health", "-o", "/dev/null"]

EXPOSE 8000

CMD ["uvicorn", "src.gateway.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "2"]

# ---------------------------------------------------------------------------
# Stage 3: admin (same base, different entrypoint)
# ---------------------------------------------------------------------------
FROM production AS admin

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD ["curl", "-sf", "http://localhost:8001/health", "-o", "/dev/null"]

EXPOSE 8001

CMD ["uvicorn", "src.admin.main:app", "--host", "0.0.0.0", "--port", "8001", "--workers", "1"]

# ---------------------------------------------------------------------------
# Stage 4: development (includes dev deps + hot reload)
# ---------------------------------------------------------------------------
FROM deps AS dev

# Install dev dependencies
RUN poetry install --no-interaction --no-ansi --no-root

# Runtime deps
RUN apt-get update && \
    apt-get install -y --no-install-recommends libpq5 curl && \
    rm -rf /var/lib/apt/lists/*

COPY src/ ./src/
COPY tests/ ./tests/
COPY scripts/ ./scripts/
COPY alembic.ini* ./
COPY pyproject.toml ./

ENV PATH="/app/.venv/bin:$PATH"
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

EXPOSE 8000 8001 5678

CMD ["uvicorn", "src.gateway.main:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]
