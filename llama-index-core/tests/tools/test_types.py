from typing import Optional, Type

import pytest

from llama_index.core.bridge.pydantic import BaseModel
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


@pytest.mark.parametrize("args_schema", [Outer, None])
def test_langchain_kwargs_preserve_explicit_schema(
    args_schema: Optional[Type[BaseModel]],
) -> None:
    tool = FunctionTool.from_defaults(lambda value: value, name="echo")

    kwargs = tool._process_langchain_tool_kwargs({"args_schema": args_schema})

    assert kwargs["args_schema"] is args_schema


def test_langchain_kwargs_default_to_tool_schema() -> None:
    tool = FunctionTool.from_defaults(lambda value: value, name="echo")

    kwargs = tool._process_langchain_tool_kwargs({})

    assert kwargs["args_schema"] is tool.metadata.fn_schema
