from typing import List, Literal, Union

import pytest

from llama_index.core.bridge.pydantic import BaseModel, RootModel
from llama_index.core.program.function_program import get_function_tool
from llama_index.core.tools.function_tool import FunctionTool
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


def test_parameterless_tool_schema() -> None:
    def ping() -> str:
        """Ping service."""
        return "pong"

    tool = FunctionTool.from_defaults(fn=ping)
    schema = tool.metadata.get_parameters_dict()
    assert schema == {
        "type": "object",
        "properties": {},
        "required": [],
    }

    openai_tool = tool.metadata.to_openai_tool()
    assert openai_tool["function"]["parameters"] == {
        "type": "object",
        "properties": {},
        "required": [],
    }


def test_all_default_args_tool_schema() -> None:
    def greet(name: str = "world") -> str:
        """Greet someone."""
        return f"hello {name}"

    tool = FunctionTool.from_defaults(fn=greet)
    schema = tool.metadata.get_parameters_dict()
    assert schema["type"] == "object"
    assert "name" in schema["properties"]
    assert schema["required"] == []


def test_root_model_tool_schema_keeps_defs_and_ref() -> None:
    class ModelInner(BaseModel):
        x: int

    class Root(RootModel[ModelInner]):
        pass

    schema = ToolMetadata(
        description="d", name="t", fn_schema=Root
    ).get_parameters_dict()
    assert "$defs" in schema
    assert "$ref" in schema
    assert schema["$ref"] == "#/$defs/ModelInner"


def test_root_model_union_tool_schema_keeps_anyof() -> None:
    class Approve(BaseModel):
        action: Literal["approve"]
        note: str

    class Reject(BaseModel):
        action: Literal["reject"]
        reason: str

    class Decision(RootModel[Union[Approve, Reject]]):
        pass

    schema = ToolMetadata(
        description="d", name="t", fn_schema=Decision
    ).get_parameters_dict()
    assert "$defs" in schema
    assert "anyOf" in schema
    assert len(schema["anyOf"]) == 2


def test_root_model_list_tool_schema_keeps_items() -> None:
    class ItemList(RootModel[List[int]]):
        pass

    schema = ToolMetadata(
        description="d", name="t", fn_schema=ItemList
    ).get_parameters_dict()
    assert schema == {"type": "array", "items": {"type": "integer"}}
