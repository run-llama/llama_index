"""Test tool spec."""

from typing import List, Tuple, Union

import pytest
from llama_index.core.bridge.pydantic import BaseModel
from llama_index.core.tools.tool_spec.base import BaseToolSpec
from llama_index.core.tools.types import ToolMetadata
from llama_index.core.workflow import Context


class FooSchema(BaseModel):
    arg1: str
    arg2: int


class BarSchema(BaseModel):
    arg1: bool


class AbcSchema(BaseModel):
    arg1: str


class TestToolSpec(BaseToolSpec):
    spec_functions: List[Union[str, Tuple[str, str]]] = [
        "foo",
        "bar",
        "abc",
        "abc_with_ctx",
        "async_only_fn",
    ]

    def foo(self, arg1: str, arg2: int) -> str:
        """Foo."""
        return f"foo {arg1} {arg2}"

    def bar(self, arg1: bool) -> str:
        """Bar."""
        return f"bar {arg1}"

    async def afoo(self, arg1: str, arg2: int) -> str:
        """Afoo."""
        return self.foo(arg1=arg1, arg2=arg2)

    async def abar(self, arg1: bool) -> str:
        """Abar."""
        return self.bar(arg1=arg1)

    async def async_only_fn(self) -> str:
        """Async only fn."""
        return "async only fn"

    def abc(self, arg1: str) -> str:
        # NOTE: no docstring
        return f"bar {arg1}"

    def abc_with_ctx(self, arg1: str, ctx: Context) -> str:
        return f"bar {arg1}"

    def unused_function(self, arg1: str) -> str:
        return f"unused {arg1}"


def test_tool_spec() -> None:
    """Test tool spec."""
    tool_spec = TestToolSpec()
    # first is foo, second is bar
    tools = tool_spec.to_tool_list()
    assert len(tools) == 5
    assert tools[0].metadata.name == "foo"
    assert tools[0].metadata.description == "foo(arg1: str, arg2: int) -> str\nFoo."
    assert tools[0].fn("hello", 1) == "foo hello 1"
    assert tools[0].ctx_param_name is None
    assert not tools[0].requires_context

    assert tools[1].metadata.name == "bar"
    assert tools[1].metadata.description == "bar(arg1: bool) -> str\nBar."
    assert str(tools[1](True)) == "bar True"
    assert tools[1].ctx_param_name is None
    assert not tools[1].requires_context

    assert tools[2].metadata.name == "abc"
    assert tools[2].metadata.description == "abc(arg1: str) -> str\n"
    assert (
        tools[2].metadata.fn_schema.model_json_schema()["properties"]
        == AbcSchema.model_json_schema()["properties"]
    )
    assert tools[2].ctx_param_name is None
    assert not tools[2].requires_context

    assert tools[3].metadata.name == "abc_with_ctx"
    assert tools[3].metadata.description == "abc_with_ctx(arg1: str) -> str\n"
    assert (
        tools[3].metadata.fn_schema.model_json_schema()["properties"]
        == AbcSchema.model_json_schema()["properties"]
    )
    assert tools[3].ctx_param_name == "ctx"
    assert tools[3].requires_context

    # test metadata mapping
    tools = tool_spec.to_tool_list(
        func_to_metadata_mapping={
            "foo": ToolMetadata(
                "foo_description", name="foo_name", fn_schema=FooSchema
            ),
        }
    )
    assert len(tools) == 5
    assert tools[0].metadata.name == "foo_name"
    assert tools[0].metadata.description == "foo_description"
    assert tools[0].metadata.fn_schema is not None
    fn_schema = tools[0].metadata.fn_schema.model_json_schema()
    print(fn_schema)
    assert fn_schema["properties"]["arg1"]["type"] == "string"
    assert fn_schema["properties"]["arg2"]["type"] == "integer"
    assert tools[1].metadata.name == "bar"
    assert tools[1].metadata.description == "bar(arg1: bool) -> str\nBar."
    assert tools[1].metadata.fn_schema is not None
    fn_schema = tools[1].metadata.fn_schema.model_json_schema()
    assert fn_schema["properties"]["arg1"]["type"] == "boolean"


@pytest.mark.asyncio
async def test_tool_spec_async() -> None:
    """Test async_fn of tool spec."""
    tool_spec = TestToolSpec()
    tools = tool_spec.to_tool_list()
    assert len(tools) == 5

    assert await tools[0].async_fn("hello", 1) == "foo hello 1"
    assert str(await tools[1].acall(True)) == "bar True"

    assert tools[0].fn("hello", 1) == "foo hello 1"
    assert str(tools[1](True)) == "bar True"


def test_async_patching() -> None:
    # test sync patching of async function
    tool_spec = TestToolSpec()
    tool_spec.spec_functions = ["afoo", "async_only_fn"]
    tools = tool_spec.to_tool_list()
    assert len(tools) == 2
    assert tools[0].fn("hello", 1) == "foo hello 1"

    assert tools[0].metadata.name == "afoo"
    assert tools[0].metadata.description == "afoo(arg1: str, arg2: int) -> str\nAfoo."
    assert tools[1].metadata.name == "async_only_fn"
    assert tools[1].metadata.description == "async_only_fn() -> str\nAsync only fn."


def test_tool_spec_subset() -> None:
    """Test tool spec subset."""
    tool_spec = TestToolSpec()
    tools = tool_spec.to_tool_list(spec_functions=["abc"])
    assert len(tools) == 1
    assert tools[0].metadata.name == "abc"
    assert tools[0].metadata.description == "abc(arg1: str) -> str\n"
    assert (
        tools[0].metadata.fn_schema.model_json_schema()["properties"]
        == AbcSchema.model_json_schema()["properties"]
    )
