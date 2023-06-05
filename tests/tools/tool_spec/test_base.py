"""Test tool spec."""

from pydantic import BaseModel
from llama_index.tools.tool_spec.base import BaseToolSpec
from llama_index.tools.types import ToolMetadata
from typing import List, Type


class FooSchema(BaseModel):
    arg1: str
    arg2: int


class BarSchema(BaseModel):
    arg1: bool


class AbcSchema(BaseModel):
    arg1: str


class TestToolSpec(BaseToolSpec):

    spec_functions: List[str] = ["foo", "bar", "abc"]

    def foo(self, arg1: str, arg2: int) -> str:
        """Foo."""
        return f"foo {arg1} {arg2}"

    def bar(self, arg1: bool) -> str:
        """Bar."""
        return f"bar {arg1}"

    def abc(self, arg1: str) -> str:
        # NOTE: no docstring
        return f"bar {arg1}"

    def get_fn_schema_from_fn_name(self, fn_name: str) -> Type[BaseModel]:
        """Return map from function name."""
        if fn_name == "foo":
            return FooSchema
        elif fn_name == "bar":
            return BarSchema
        elif fn_name == "abc":
            return AbcSchema
        else:
            raise ValueError(f"Invalid function name: {fn_name}")


def test_tool_spec() -> None:
    """Test tool spec."""
    tool_spec = TestToolSpec()
    # first is foo, second is bar
    tools = tool_spec.to_tool_list()
    assert len(tools) == 3
    assert tools[0].metadata.name == "foo"
    assert tools[0].metadata.description == "foo(arg1: str, arg2: int) -> str\nFoo."
    assert tools[0].fn("hello", 1) == "foo hello 1"
    assert tools[1].metadata.name == "bar"
    assert tools[1].metadata.description == "bar(arg1: bool) -> str\nBar."
    assert tools[1](True) == "bar True"
    assert tools[2].metadata.name == "abc"
    assert tools[2].metadata.description == "abc(arg1: str) -> str\n"
    assert tools[2].metadata.fn_schema == AbcSchema

    # test metadata mapping
    tools = tool_spec.to_tool_list(
        func_to_metadata_mapping={
            "foo": ToolMetadata("foo_description", name="foo_name"),
        }
    )
    assert len(tools) == 3
    assert tools[0].metadata.name == "foo_name"
    assert tools[0].metadata.description == "foo_description"
    assert tools[0].metadata.fn_schema is not None
    fn_schema = tools[0].metadata.fn_schema.schema()
    assert fn_schema["properties"]["arg1"]["type"] == "string"
    assert fn_schema["properties"]["arg2"]["type"] == "integer"
    assert tools[1].metadata.name == "bar"
    assert tools[1].metadata.description == "bar(arg1: bool) -> str\nBar."
    assert tools[1].metadata.fn_schema is not None
    fn_schema = tools[1].metadata.fn_schema.schema()
    assert fn_schema["properties"]["arg1"]["type"] == "boolean"


def test_tool_spec_schema() -> None:
    """Test tool spec schemas match."""
    tool_spec = TestToolSpec()
    # first is foo, second is bar
    schema1 = tool_spec.get_fn_schema_from_fn_name("foo")
    assert schema1 == FooSchema
    schema2 = tool_spec.get_fn_schema_from_fn_name("bar")
    assert schema2 == BarSchema
