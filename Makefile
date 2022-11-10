.PHONY: format lint

format:
	black .
	isort .

lint:
	mypy .
	black . --check
	isort . --check
	flake8 .