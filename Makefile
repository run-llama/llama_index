GIT_ROOT ?= $(shell git rev-parse --show-toplevel)
help: ## Show all Makefile targets
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[33m%-30s\033[0m %s\n", $$1, $$2}'

.PHONY: format lint
format: ## Run code formatter: black
	black .
lint: ## Run linters: mypy, black, ruff
	mypy .
	black . --check
	ruff check .
test: ## Run tests
	pytest tests
watch-docs: ## Build and watch documentation
	sphinx-autobuild docs/ docs/_build/html --open-browser --watch $(GIT_ROOT)/llama_index/
