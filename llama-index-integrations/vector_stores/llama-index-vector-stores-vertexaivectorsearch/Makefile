GIT_ROOT ?= $(shell git rev-parse --show-toplevel)

help:	## Show all Makefile targets.
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[33m%-30s\033[0m %s\n", $$1, $$2}'

format:	## Run code autoformatters (ruff).
	pre-commit install
	git ls-files | xargs pre-commit run ruff-format --files

lint:	## Run linters: pre-commit (black, ruff, codespell) and mypy
	pre-commit install && git ls-files | xargs pre-commit run --show-diff-on-failure --files

test:	## Run tests via pytest.
	uv run --extra=v2 pytest tests

.PHONY: testcov
testcov:	## Run tests (with coverage) via pytest and tox.
	# tests multiple versions and extra dependencies (e.g., 'v2')
	uv run tox run-parallel

watch-docs:	## Build and watch documentation.
	sphinx-autobuild docs/ docs/_build/html --open-browser --watch $(GIT_ROOT)/llama_index/
