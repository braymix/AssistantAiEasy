.PHONY: help install dev dev-admin test test-fast lint format \
       up up-enterprise up-dev down down-enterprise logs shell \
       build clean init migrate seed pull-model

PROFILE ?= mini

# ============================================================================
# Help
# ============================================================================

help: ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'

# ============================================================================
# Development (local)
# ============================================================================

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

test-fast: ## Run tests without coverage (stop on first failure)
	poetry run pytest -x -q

lint: ## Run linter and type checker
	poetry run ruff check src/ tests/
	poetry run mypy src/

format: ## Format code with ruff
	poetry run ruff format src/ tests/
	poetry run ruff check --fix src/ tests/

# ============================================================================
# Database
# ============================================================================

init: ## Initialize database
	KNOWLEDGEHUB_PROFILE=$(PROFILE) poetry run python scripts/init_db.py

migrate: ## Run database migrations
	KNOWLEDGEHUB_PROFILE=$(PROFILE) poetry run python scripts/migrate.py

seed: ## Seed initial detection rules
	KNOWLEDGEHUB_PROFILE=$(PROFILE) poetry run python scripts/seed_rules.py

# ============================================================================
# Docker – Mini (default)
# ============================================================================

build: ## Build Docker images
	docker compose build

up: ## Start mini stack (gateway + admin + ollama + chroma + open-webui)
	docker compose up -d

up-dev: ## Start mini stack in dev mode (hot reload + debug ports)
	docker compose -f docker-compose.yml -f docker-compose.dev.yml up -d --build

down: ## Stop mini stack and remove containers
	docker compose down

# ============================================================================
# Docker – Enterprise
# ============================================================================

up-enterprise: ## Start enterprise stack (+ postgres, qdrant, redis, vllm)
	docker compose -f docker-compose.yml -f docker-compose.enterprise.yml up -d

down-enterprise: ## Stop enterprise stack
	docker compose -f docker-compose.yml -f docker-compose.enterprise.yml down

# ============================================================================
# Docker – Operations
# ============================================================================

logs: ## Tail logs for all services (SERVICE= to filter)
ifdef SERVICE
	docker compose logs -f $(SERVICE)
else
	docker compose logs -f
endif

shell: ## Open a shell in a service container (SERVICE=gateway)
	docker compose exec $(or $(SERVICE),gateway) /bin/bash

ps: ## Show running containers
	docker compose ps

restart: ## Restart a service (SERVICE=gateway)
	docker compose restart $(or $(SERVICE),gateway)

pull-model: ## Pull an Ollama model (MODEL=llama3.2)
	docker compose exec ollama ollama pull $(or $(MODEL),llama3.2)

# ============================================================================
# Cleanup
# ============================================================================

clean: ## Remove cached files and build artifacts
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true
	rm -rf .pytest_cache .mypy_cache .ruff_cache htmlcov .coverage

clean-docker: ## Remove all containers, volumes, and images
	docker compose down -v --rmi local
