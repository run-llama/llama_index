.PHONY: install setup lint format test mypy clean

# Setup and Dependencies
setup:
	poetry install --with dev

# Linting and Formatting
lint:
	ruff check .
format:
	ruff check . --fix
	ruff format .

# Testing and Typing
test:
	pytest tests/
mypy:
	mypy .

# Clean up build artifacts
clean:
	find . -name "__pycache__" -exec rm -rf {} +
	rm -rf .pytest_cache
	rm -rf .mypy_cache
	rm -rf dist
	rm -rf *.egg-info