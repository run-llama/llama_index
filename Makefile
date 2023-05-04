.PHONY: format lint

GIT_ROOT ?= $(shell git rev-parse --show-toplevel)

format:
	black .

lint:
	mypy .
	black . --check
	ruff check .

test:
	pytest tests

# Docs
watch-docs: ## Build and watch documentation
	sphinx-autobuild docs/ docs/_build/html --open-browser --watch $(GIT_ROOT)/llama_index/
