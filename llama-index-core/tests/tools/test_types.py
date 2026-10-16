import pytest

from llama_index.core.bridge.pydantic import BaseModel
from llama_index.core.program.function_program import get_function_tool
from llama_index.core.tools import FunctionTool
from llama_index.core.tools.types import ToolMetadata


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


def test_zero_parameter_tool_schema_declares_required() -> None:
    def ping() -> str:
        """Ping the service."""
        return "pong"

    tool = FunctionTool.from_defaults(fn=ping)

    assert tool.metadata.get_parameters_dict() == {
        "type": "object",
        "properties": {},
        "required": [],
    }
    parameters = tool.metadata.to_openai_tool()["function"]["parameters"]
    assert parameters["required"] == []


def test_all_optional_parameters_tool_schema_declares_required() -> None:
    def greet(name: str = "world") -> str:
        """Greet someone."""
        return f"hello {name}"

    tool = FunctionTool.from_defaults(fn=greet)

    schema = tool.metadata.get_parameters_dict()
    assert schema["required"] == []
    assert set(schema["properties"]) == {"name"}
