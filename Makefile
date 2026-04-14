# QE Makefile

# Installation
install:
	poetry install
	poetry run pre-commit install --allow-missing-config -f
	poetry run detect-secrets scan > .secrets.baseline

# Testing
test:
	poetry run pytest
	@rm -f .coverage.* || true
	@echo "===== Finished running unit tests ====="

integration_test:
	poetry run pytest tests/integration
	@echo "===== Finished running integration tests ====="

# Linting and formatting
lint:
	poetry run ruff check .

lint-fix:
	poetry run ruff check --fix .

format:
	poetry run ruff format .

format-check:
	poetry run ruff format --check .

# Type checking
typecheck:
	poetry run mypy qe

# Security
security:
	poetry run bandit -c pyproject.toml -r qe

# Pre-commit
pre-commit:
	poetry run pre-commit run --all-files

# All checks (for CI)
check: lint-fix format typecheck security test

# Qdrant database
qdrant-dump:
	docker run --rm \
		-v qdrant_data:/qdrant/storage:ro \
		-v $(PWD):/backup \
		busybox \
		tar czf /backup/qdrant_storage_dump.tar.gz -C /qdrant/storage .
	@echo "===== Qdrant database dumped to qdrant_storage_dump.tar.gz ====="

# PostgreSQL database
pg-dump:
	docker exec qe-postgres pg_dump -U qe qe > qe_postgres_dump.sql
	@echo "===== PostgreSQL database dumped to qe_postgres_dump.sql ====="

# Clean
clean:
	rm -rf .coverage .coverage.* coverage.json lcov.info htmlcov
	rm -rf .pytest_cache .mypy_cache .ruff_cache
	rm -rf dist build *.egg-info
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true

# Development server
dev:
	poetry run uvicorn api.main:app --reload

.PHONY: install test integration_test lint lint-fix format format-check typecheck security pre-commit check clean qdrant-dump dev
