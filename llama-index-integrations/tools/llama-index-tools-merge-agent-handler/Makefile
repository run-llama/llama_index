lint:
	uv run --extra dev ruff check .
	uv run --extra dev mypy llama_index/tools/merge_agent_handler/base.py tests/test_tools.py examples/merge_agent_handler_example.py ui/merge_agent_handler_ui.py

test:
	uv run --extra dev pytest tests/ --cov=llama_index/tools/merge_agent_handler --cov-report=term-missing
