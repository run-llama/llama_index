.PHONY: format lint test clean

format:
	black .
	ruff check --fix .

lint:
	ruff check .
	mypy llama_index/tools/serpex

test:
	pytest tests/ -v

test-coverage:
	pytest --cov=llama_index.tools.serpex tests/ -v

clean:
	rm -rf build/ dist/ *.egg-info
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name '*.pyc' -delete
