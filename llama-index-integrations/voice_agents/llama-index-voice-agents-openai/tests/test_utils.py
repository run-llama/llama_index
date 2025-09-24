import pytest

from llama_index.voice_agents.openai.utils import get_tool_by_name
from llama_index.core.tools import FunctionTool
from typing import List


@pytest.fixture()
def function_tools() -> List[FunctionTool]:
    def add(i: int, j: int) -> int:
        return i + j

    def greet(name: str) -> str:
        return "Hello " + name

    async def hello_world() -> str:
        return "Hello World!"

    return [
        FunctionTool.from_defaults(fn=add, name="add_tool"),
        FunctionTool.from_defaults(fn=greet, name="greet_tool"),
        FunctionTool.from_defaults(fn=hello_world, name="hello_world_tool"),
    ]


def test_tools_from_names(function_tools: List[FunctionTool]) -> None:
    tool = get_tool_by_name(function_tools, name="add_tool")
    assert tool.metadata.get_name() == "add_tool"
    assert tool(**{"i": 5, "j": 7}).raw_output == 12
    tool1 = get_tool_by_name(function_tools, name="greet_tool")
    assert tool1.metadata.get_name() == "greet_tool"
    assert tool1(**{"name": "Mark"}).raw_output == "Hello Mark"
    tool2 = get_tool_by_name(function_tools, name="hello_world_tool")
    assert tool2.metadata.get_name() == "hello_world_tool"
    assert tool2().raw_output == "Hello World!"
    assert get_tool_by_name(function_tools, name="test_tool") is None
