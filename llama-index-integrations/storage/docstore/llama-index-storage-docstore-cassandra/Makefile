.PHONY: test lint format typecheck

test:
	pytest tests/ -v

lint:
	ruff check llama_index/ tests/

format:
	ruff format llama_index/ tests/

typecheck:
	mypy llama_index/