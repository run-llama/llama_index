from llama_index.tools.gestell import GestellToolSpec
from llama_index.core.tools.tool_spec.base import FunctionTool


def test_to_tool_list_returns_expected_tools():
    """``to_tool_list`` should return the expected tools with correct names and functions."""
    spec = GestellToolSpec()
    tools = spec.to_tool_list()
    assert isinstance(tools, list)
    assert len(tools) == 2
    assert all(isinstance(tool, FunctionTool) for tool in tools)

    tool_map = {tool.metadata.name: tool for tool in tools}

    assert set(tool_map.keys()) == {"gestell_search", "gestell_prompt"}
