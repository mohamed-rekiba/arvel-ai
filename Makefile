# arvel-ai — developer tasks. Gate targets mirror CI + `make check`.
RUN ?= uv run

.DEFAULT_GOAL := help
.PHONY: help sync lint format format-check typecheck imports test test-integration check pre-commit hooks clean

help:  ## Show this help
	@grep -hE '^[a-zA-Z_-]+:.*?## ' $(MAKEFILE_LIST) | \
		awk 'BEGIN{FS=":.*?## "}{printf "  \033[36m%-16s\033[0m %s\n", $$1, $$2}'

sync:  ## Create/refresh the dev environment (all extras + dev tools)
	uv sync --all-extras

lint:  ## Ruff lint
	$(RUN) ruff check .

format:  ## Ruff format (writes)
	$(RUN) ruff format .

format-check:  ## Ruff format (check only)
	$(RUN) ruff format --check .

typecheck:  ## Strict mypy
	$(RUN) mypy src

imports:  ## import-linter — keeps engines (litellm/httpx/temporalio) off the import path
	PYTHONPATH=src $(RUN) lint-imports

test:  ## pytest (hermetic; integration tiers are env-gated)
	$(RUN) pytest

test-integration:  ## real-service tiers (needs Docker / provider keys / AI_TEMPORAL_TARGET)
	docker compose -f docker-compose.test.yml up -d
	AI_TEMPORAL_TARGET=localhost:7233 $(RUN) pytest tests/test_workflow_temporal.py

check: lint format-check typecheck imports test  ## Everything CI runs

pre-commit:  ## Run all pre-commit hooks across the repo
	$(RUN) pre-commit run --all-files

hooks:  ## Install pre-commit git hooks
	$(RUN) pre-commit install

clean:  ## Remove caches + build artifacts
	rm -rf .pytest_cache .mypy_cache .ruff_cache dist build **/__pycache__
