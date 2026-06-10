GIT_ROOT ?= $(shell git rev-parse --show-toplevel)

help:
	@echo "Available targets:"
	@echo "  install       Install package in editable mode"
	@echo "  fmt           Run ruff formatter"
	@echo "  lint          Run ruff linter"
	@echo "  test          Run unit tests"

install:
	pip install -e .

fmt:
	ruff format .

lint:
	ruff check .

test:
	pytest tests/ -v
