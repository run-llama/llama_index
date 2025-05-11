import pytest
from llama_index.tools.gestell import GestellToolSpec
from llama_index.core.tools.tool_spec.base import FunctionTool


@pytest.mark.asyncio
async def test_asearch_returns_list():
    """`asearch` should return a list, and if non-empty, mapping-like items."""
    spec = GestellToolSpec()
    results = await spec.asearch(query="example query")
    assert isinstance(results, list)


@pytest.mark.asyncio
async def test_aprompt_returns_non_empty_string():
    """`aprompt` should return a non-empty string containing response text."""
    spec = GestellToolSpec()
    response = await spec.aprompt(prompt="Hello world")
    assert isinstance(response, str)
    assert response.strip() != ""


def test_to_tool_list_returns_two_functiontools():
    """`to_tool_list` should return exactly two FunctionTool instances."""
    spec = GestellToolSpec()
    tools = spec.to_tool_list()
    assert isinstance(tools, list)
    assert len(tools) == 2
    assert all(isinstance(tool, FunctionTool) for tool in tools)
