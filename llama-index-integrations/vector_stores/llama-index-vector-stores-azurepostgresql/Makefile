create-env:
	@echo "Creating virtual environment..."
	uv sync
	@echo "Virtual environment created."

lint lint_package lint_tests: create-env
	@echo "Running linters..."
	uv run tox run -e lint || true
	@echo "Linters completed."

test: create-env
	@echo "Running tests..."
	uv run pytest
	@echo "Tests completed."

.PHONY: create-env lint test
