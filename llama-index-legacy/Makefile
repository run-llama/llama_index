GIT_ROOT ?= $(shell git rev-parse --show-toplevel)

help:	## Show all Makefile targets.
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[33m%-30s\033[0m %s\n", $$1, $$2}'

format:	## Run code autoformatters (black).
	pre-commit install
	pre-commit run black --all-files

lint:	## Run linters: pre-commit (black, ruff, codespell) and mypy
	pre-commit install && pre-commit run --all-files --show-diff-on-failure

test:	## Run tests via pytest.
	pytest tests

watch-docs:	## Build and watch documentation.
	sphinx-autobuild docs/ docs/_build/html --open-browser --watch $(GIT_ROOT)/llama_index/
