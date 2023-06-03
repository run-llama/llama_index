"""Test tools."""

from llama_index.tools.function_tool import FunctionTool
from llama_index.tools.types import BaseTool
from pydantic import BaseModel


def test_function_tool() -> None:
    """Test function tool."""

    function_tool = FunctionTool.from_defaults(
        lambda x: str(x), name="foo", description="bar"
    )
    assert function_tool.metadata.name == "foo"
    assert function_tool.metadata.description == "bar"

    result = function_tool(1)
    assert result == "1"

    # test to langchain
    # NOTE: can't take in a function with int args
    langchain_tool = function_tool.to_langchain_tool()
    result = langchain_tool.run("1")
    assert result == "1"

    # test langchain structured tool
    class TestSchema(BaseModel):
        x: int
        y: int

    function_tool = FunctionTool.from_defaults(
        lambda x, y: str(x) + "," + str(y),
        name="foo",
        description="bar",
        fn_schema=TestSchema,
    )
    assert function_tool(1, 2) == "1,2"
    langchain_tool = function_tool.to_langchain_structured_tool()
    assert langchain_tool.run({"x": 1, "y": 2}) == "1,2"
    assert langchain_tool.args_schema == TestSchema
