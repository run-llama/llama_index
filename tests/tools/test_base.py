"""Test tools."""
import pytest

try:
    from pydantic.v1 import BaseModel
except ImportError:
    from pydantic import BaseModel

from llama_index.tools.function_tool import FunctionTool


def tmp_function(x: int) -> str:
    return str(x)


async def async_tmp_function(x: int) -> str:
    return "async_" + str(x)


def test_function_tool() -> None:
    """Test function tool."""

    function_tool = FunctionTool.from_defaults(lambda x: str(x), name="foo", description="bar")
    assert function_tool.metadata.name == "foo"
    assert function_tool.metadata.description == "bar"
    assert function_tool.metadata.fn_schema is not None
    actual_schema = function_tool.metadata.fn_schema.schema()
    # note: no type
    assert "x" in actual_schema["properties"]

    result = function_tool(1)
    assert str(result) == "1"

    # test adding typing to function

    function_tool = FunctionTool.from_defaults(tmp_function, name="foo", description="bar")
    assert function_tool.metadata.fn_schema is not None
    actual_schema = function_tool.metadata.fn_schema.schema()
    assert actual_schema["properties"]["x"]["type"] == "integer"

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
    assert str(function_tool(1, 2)) == "1,2"
    langchain_tool2 = function_tool.to_langchain_structured_tool()
    assert langchain_tool2.run({"x": 1, "y": 2}) == "1,2"
    assert langchain_tool2.args_schema == TestSchema


@pytest.mark.asyncio
async def test_function_tool_async() -> None:
    """Test function tool async."""
    function_tool = FunctionTool.from_defaults(
        fn=tmp_function, async_fn=async_tmp_function, name="foo", description="bar"
    )
    assert function_tool.metadata.fn_schema is not None
    actual_schema = function_tool.metadata.fn_schema.schema()
    assert actual_schema["properties"]["x"]["type"] == "integer"

    assert str(function_tool(2)) == "2"
    assert str(await function_tool.acall(2)) == "async_2"

    # test to langchain
    # NOTE: can't take in a function with int args
    langchain_tool = function_tool.to_langchain_tool()
    result = await langchain_tool.arun("1")
    assert result == "async_1"

    # test langchain structured tool
    class TestSchema(BaseModel):
        x: int
        y: int

    def structured_tmp_function(x: int, y: int) -> str:
        return str(x) + "," + str(y)

    async def async_structured_tmp_function(x: int, y: int) -> str:
        return "async_" + str(x) + "," + str(y)

    function_tool = FunctionTool.from_defaults(
        fn=structured_tmp_function,
        async_fn=async_structured_tmp_function,
        name="foo",
        description="bar",
        fn_schema=TestSchema,
    )
    assert str(await function_tool.acall(1, 2)) == "async_1,2"
    langchain_tool2 = function_tool.to_langchain_structured_tool()
    assert (await langchain_tool2.arun({"x": 1, "y": 2})) == "async_1,2"
    assert langchain_tool2.args_schema == TestSchema


@pytest.mark.asyncio
async def test_function_tool_async_defaults() -> None:
    """Test async calls to function tool when only sync function is given."""
    function_tool = FunctionTool.from_defaults(fn=tmp_function, name="foo", description="bar")
    assert function_tool.metadata.fn_schema is not None
    actual_schema = function_tool.metadata.fn_schema.schema()
    assert actual_schema["properties"]["x"]["type"] == "integer"

    # test to langchain
    # NOTE: can't take in a function with int args
    langchain_tool = function_tool.to_langchain_tool()
    result = await langchain_tool.arun("1")
    assert result == "1"
