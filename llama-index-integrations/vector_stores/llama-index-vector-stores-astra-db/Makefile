GIT_ROOT ?= $(shell git rev-parse --show-toplevel)

help:	## Show all Makefile targets.
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[33m%-30s\033[0m %s\n", $$1, $$2}'

format:	## Run code autoformatters (black).
	poetry run pre-commit install
	git ls-files | xargs poetry run pre-commit run black --files

lint:	## Run linters: pre-commit (black, ruff, codespell) and mypy
	poetry run pre-commit install && git ls-files | xargs poetry run pre-commit run --show-diff-on-failure --files

typecheck:
	poetry run mypy --explicit-package-bases llama_index

test:	## Run tests via pytest.
	poetry run pytest tests

watch-docs:	## Build and watch documentation.
	sphinx-autobuild docs/ docs/_build/html --open-browser --watch $(GIT_ROOT)/llama_index/
