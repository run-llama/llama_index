GIT_ROOT ?= $(shell git rev-parse --show-toplevel)

help:	## Show all Makefile targets.
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[33m%-30s\033[0m %s\n", $$1, $$2}'

format:	## Run code autoformatters (black).
	pre-commit install
	git ls-files | xargs pre-commit run black --files

lint:	## Run linters: pre-commit (black, ruff, codespell) and mypy
	pre-commit install && git ls-files | xargs pre-commit run --show-diff-on-failure --files

test:	## Run tests via pants
	pants --level=error --no-local-cache --changed-since=origin/main --changed-dependents=transitive --no-test-use-coverage test

test-core:	## Run tests via pants
	pants --no-local-cache test llama-index-core/::

test-integrations:	## Run tests via pants
	pants --no-local-cache test llama-index-integrations/::

test-finetuning:	## Run tests via pants
	pants --no-local-cache test llama-index-finetuning/::

test-experimental:	## Run tests via pants
	pants --no-local-cache test llama-index-experimental/::

test-packs:	## Run tests via pants
	pants --no-local-cache test llama-index-packs/::

watch-docs:	## Build and watch documentation.
	sphinx-autobuild docs/ docs/_build/html --open-browser --watch $(GIT_ROOT)/llama_index/
