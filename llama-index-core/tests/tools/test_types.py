import pytest

from llama_index.core.bridge.pydantic import BaseModel
from llama_index.core.program.function_program import get_function_tool
from llama_index.core.tools.function_tool import FunctionTool
from llama_index.core.tools.types import ToolMetadata, ToolOutput


class Inner(BaseModel):
    name: str


class Outer(BaseModel):
    inner: Inner


def test_toolmetadata_openai_tool_description_max_length() -> None:
    openai_tool_description_limit = 1024
    valid_description = "a" * openai_tool_description_limit
    invalid_description = "a" * (1 + openai_tool_description_limit)

    ToolMetadata(valid_description).to_openai_tool()
    ToolMetadata(invalid_description).to_openai_tool(skip_length_check=True)

    with pytest.raises(ValueError):
        ToolMetadata(invalid_description).to_openai_tool()


def test_nested_tool_schema() -> None:
    tool = get_function_tool(Outer)
    schema = tool.metadata.get_parameters_dict()

    assert "$defs" in schema
    defs = schema["$defs"]
    assert "Inner" in defs
    inner = defs["Inner"]
    assert inner["required"][0] == "name"
    assert inner["properties"] == {"name": {"title": "Name", "type": "string"}}

    assert schema["required"][0] == "inner"
    assert schema["properties"] == {"inner": {"$ref": "#/$defs/Inner"}}


def test_tool_output_raw_input_defaults_to_empty_dict() -> None:
    tool_output = ToolOutput(tool_name="test_tool", content="hello")

    assert tool_output.raw_input == {}


def test_tool_output_accepts_raw_input_none() -> None:
    tool_output = ToolOutput(tool_name="test_tool", content="hello", raw_input=None)

    assert tool_output.raw_input == {}


def test_function_tool_callback_returning_tool_output_without_raw_input() -> None:
    def fn(x: int) -> str:
        return str(x)

    def callback(result: str) -> ToolOutput:
        return ToolOutput(tool_name="test_tool", content=f"overridden: {result}")

    tool = FunctionTool.from_defaults(fn=fn, callback=callback)
    output = tool(5)

    assert output.content == "overridden: 5"
