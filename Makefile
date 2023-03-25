.PHONY: format lint

format:
	black .

lint:
	mypy .
	black . --check
	ruff check .

test:
	pytest tests
