GIT_ROOT ?= $(shell git rev-parse --show-toplevel)

help:
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[33m%-30s\033[0m %s\n", $$1, $$2}'

format:
	ruff check --fix .
	ruff format .

lint:
	ruff check .
	ruff format --check .
	mypy .

test:
	pytest tests
