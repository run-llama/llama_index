help:	## Show all Makefile targets.
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-30s\033[0m %s\n", $$1, $$2}'

format:	## Run code autoformatters (ruff).
	ruff format .

lint:	## Run linters: ruff.
	ruff check .

test:	## Run tests via pytest.
	pytest tests

.PHONY: help format lint test
