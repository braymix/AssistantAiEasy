FROM python:3.11-slim AS base

WORKDIR /app

RUN pip install --no-cache-dir poetry && \
    poetry config virtualenvs.create false

COPY pyproject.toml poetry.lock* ./
RUN poetry install --no-interaction --no-ansi --only main

COPY src/ ./src/
COPY scripts/ ./scripts/

# Production target
FROM base AS prod
CMD ["uvicorn", "src.gateway.main:app", "--host", "0.0.0.0", "--port", "8000"]

# Development target
FROM base AS dev
RUN poetry install --no-interaction --no-ansi
CMD ["uvicorn", "src.gateway.main:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]
