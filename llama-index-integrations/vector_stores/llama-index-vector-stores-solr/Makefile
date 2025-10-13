GIT_ROOT ?= $(shell git rev-parse --show-toplevel)

help:	## Show all Makefile targets.
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[33m%-30s\033[0m %s\n", $$1, $$2}'

format:	## Run code autoformatters (black).
	pre-commit install
	git ls-files | xargs pre-commit run black --files

lint:	## Run linters: pre-commit (black, ruff, codespell) and mypy
	pre-commit install && git ls-files | xargs pre-commit run --show-diff-on-failure --files

test:	## Run tests via pytest.
	pytest tests

test-unit:  ## Run only tests without docker dependencies.
	pytest -m "not uses_docker" tests

testcov:  ## Run tests and output a coverage report.
	pytest --cov=llama_index.vector_stores.solr --cov-config=pyproject.toml --cov-report=xml:coverage.xml --cov-report=term-missing --cov-branch tests

watch-docs:	## Build and watch documentation.
	sphinx-autobuild docs/ docs/_build/html --open-browser --watch $(GIT_ROOT)/llama_index/
