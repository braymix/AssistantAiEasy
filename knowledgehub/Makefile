.PHONY: help install dev test lint format up down logs clean init migrate seed

PROFILE ?= mini

help: ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'

# ---------------------------------------------------------------------------
# Development
# ---------------------------------------------------------------------------

install: ## Install dependencies with poetry
	poetry install

dev: ## Run gateway in dev mode with hot reload
	KNOWLEDGEHUB_PROFILE=$(PROFILE) poetry run uvicorn src.gateway.main:app \
		--host 0.0.0.0 --port 8000 --reload

dev-admin: ## Run admin dashboard in dev mode
	KNOWLEDGEHUB_PROFILE=$(PROFILE) poetry run uvicorn src.admin.main:app \
		--host 0.0.0.0 --port 8001 --reload

test: ## Run tests with coverage
	poetry run pytest --cov=src --cov-report=term-missing -v

test-fast: ## Run tests without coverage
	poetry run pytest -x -q

lint: ## Run linter
	poetry run ruff check src/ tests/
	poetry run mypy src/

format: ## Format code
	poetry run ruff format src/ tests/
	poetry run ruff check --fix src/ tests/

# ---------------------------------------------------------------------------
# Database
# ---------------------------------------------------------------------------

init: ## Initialize database
	KNOWLEDGEHUB_PROFILE=$(PROFILE) poetry run python scripts/init_db.py

migrate: ## Run database migrations
	KNOWLEDGEHUB_PROFILE=$(PROFILE) poetry run python scripts/migrate.py

seed: ## Seed initial detection rules
	KNOWLEDGEHUB_PROFILE=$(PROFILE) poetry run python scripts/seed_rules.py

# ---------------------------------------------------------------------------
# Docker
# ---------------------------------------------------------------------------

up: ## Start services (PROFILE=mini|full)
	docker compose --profile $(PROFILE) up -d

up-dev: ## Start services in dev mode
	docker compose --profile $(PROFILE) -f docker-compose.yml -f docker-compose.dev.yml up -d

down: ## Stop services
	docker compose --profile $(PROFILE) down

logs: ## Tail service logs
	docker compose logs -f

build: ## Build Docker images
	docker compose build

# ---------------------------------------------------------------------------
# Cleanup
# ---------------------------------------------------------------------------

clean: ## Remove cached files and build artifacts
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true
	rm -rf .pytest_cache .mypy_cache .ruff_cache htmlcov .coverage
