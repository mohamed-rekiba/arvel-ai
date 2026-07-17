# arvel-ai — developer tasks. Gate targets mirror CI + `make check`.
RUN ?= uv run

.DEFAULT_GOAL := help
.PHONY: help sync lint format format-check typecheck imports test test-live check pre-commit hooks clean

help:  ## Show this help
	@grep -hE '^[a-zA-Z_-]+:.*?## ' $(MAKEFILE_LIST) | \
		awk 'BEGIN{FS=":.*?## "}{printf "  \033[36m%-16s\033[0m %s\n", $$1, $$2}'

sync:  ## Create/refresh the dev environment (dev tools; the litellm extra is faked in tests)
	uv sync

lint:  ## Ruff lint
	$(RUN) ruff check .

format:  ## Ruff format (writes)
	$(RUN) ruff format .

format-check:  ## Ruff format (check only)
	$(RUN) ruff format --check .

typecheck:  ## Strict mypy + pyright
	$(RUN) mypy src
	$(RUN) pyright

imports:  ## import-linter — keeps the engines (litellm/httpx) off the import path
	PYTHONPATH=src $(RUN) lint-imports

test:  ## pytest (hermetic; the live-provider tier is env-gated)
	$(RUN) pytest

test-live:  ## real-provider tier — talks to an actual model (needs AI_LIVE_MODEL + a key)
	$(RUN) pytest tests/test_live_providers.py

check: lint format-check typecheck imports test  ## Everything CI runs

pre-commit:  ## Run all pre-commit hooks across the repo
	$(RUN) pre-commit run --all-files

hooks:  ## Install pre-commit git hooks
	$(RUN) pre-commit install

clean:  ## Remove caches + build artifacts
	rm -rf .pytest_cache .mypy_cache .ruff_cache dist build **/__pycache__
