.PHONY: install
install:
	pip install -e .

.PHONY: install-dev
install-dev:
	pip install -e ".[dev]"

.PHONY: format
format:
	black llama_index tests
	ruff check --fix llama_index tests

.PHONY: lint
lint:
	ruff check llama_index tests
	mypy llama_index

.PHONY: test
test:
	pytest tests/ -v

.PHONY: test-cov
test-cov:
	pytest tests/ --cov=llama_index --cov-report=term-missing

.PHONY: clean
clean:
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info
