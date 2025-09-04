.PHONY: format lint test

format:
	poetry run black .
	poetry run isort .

lint:
	poetry run mypy .
	poetry run black . --check
	poetry run isort . --check
	poetry run flake8 .

test:
	poetry run pytest tests/ --disable-socket

clean:
	rm -rf dist/*
	rm -rf build/*
	rm -rf *.egg-info

build:
	poetry build

install:
	poetry install --with dev

install_editable:
	pip install -e ".[dev]"
