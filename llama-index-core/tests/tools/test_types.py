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


class AllOptional(BaseModel):
    """Every field has a default, so pydantic emits no "required" at all."""

    query: str = ""
    limit: int = 5


class AllOptionalOuter(BaseModel):
    inner: AllOptional


def _ping() -> str:
    """Ping the service."""
    return "pong"


def _search(query: str = "", limit: int = 5) -> str:
    """Search with every argument optional."""
    return query


def test_no_arg_tool_declares_required() -> None:
    """
    A zero-argument tool must still emit a "required" list (#18928).

    Strict OpenAI-compatible servers (vLLM) validate the parameter schema and
    reject a tool whose parameters omit "required"; the request then fails
    before the model is ever called.
    """
    schema = FunctionTool.from_defaults(fn=_ping).metadata.get_parameters_dict()

    assert schema["type"] == "object"
    assert schema["properties"] == {}
    assert schema["required"] == []


def test_no_arg_tool_openai_tool_declares_required() -> None:
    """The serialized OpenAI tool carries the same "required" key."""
    openai_tool = FunctionTool.from_defaults(fn=_ping).metadata.to_openai_tool()
    params = openai_tool["function"]["parameters"]

    assert params["required"] == []
    assert params["properties"] == {}


def test_all_optional_tool_declares_required() -> None:
    """All-optional arguments must also emit a "required" list (#18928)."""
    schema = FunctionTool.from_defaults(fn=_search).metadata.get_parameters_dict()

    assert schema["required"] == []
    assert set(schema["properties"]) == {"query", "limit"}


def test_nested_all_optional_model_declares_required() -> None:
    """
    Nested models under "$defs" get the same treatment.

    pydantic inlines nested models in "$defs" and strict servers validate
    those subschemas too, so an all-optional nested model must not omit
    "required" either.
    """
    schema = get_function_tool(AllOptionalOuter).metadata.get_parameters_dict()

    assert schema["required"] == ["inner"]
    nested = schema["$defs"]["AllOptional"]
    assert nested["type"] == "object"
    assert nested["required"] == []


def test_tool_with_required_args_is_unchanged() -> None:
    """The fix must not overwrite a real "required" list."""

    class NeedsOne(BaseModel):
        a: str
        b: int = 1

    schema = get_function_tool(NeedsOne).metadata.get_parameters_dict()

    assert schema["required"] == ["a"]


def test_deprecated_openai_function_declares_required() -> None:
    """`to_openai_function` goes through the same code path."""
    with pytest.warns(DeprecationWarning):
        fn = FunctionTool.from_defaults(fn=_ping).metadata.to_openai_function()

    assert fn["parameters"]["required"] == []


def test_parameters_dict_is_json_serializable_without_required_gap() -> None:
    """Every emitted schema survives a JSON round-trip with "required" set."""
    import json

    for fn in (_ping, _search):
        params = FunctionTool.from_defaults(fn=fn).metadata.get_parameters_dict()
        assert json.loads(json.dumps(params))["required"] == []
