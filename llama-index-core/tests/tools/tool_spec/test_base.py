"""Test tool spec."""

from typing import List, Optional, Tuple, Type, Union

import pytest
from llama_index.core.bridge.pydantic import BaseModel
from llama_index.core.tools.tool_spec.base import BaseToolSpec
from llama_index.core.tools.types import ToolMetadata


class FooSchema(BaseModel):
    arg1: str
    arg2: int


class BarSchema(BaseModel):
    arg1: bool


class AbcSchema(BaseModel):
    arg1: str


class TestToolSpec(BaseToolSpec):
    spec_functions: List[Union[str, Tuple[str, str]]] = ["foo", "bar", "abc"]

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

    def abc(self, arg1: str) -> str:
        # NOTE: no docstring
        return f"bar {arg1}"

    def get_fn_schema_from_fn_name(
        self,
        fn_name: str,
        spec_functions: Optional[List[Union[str, Tuple[str, str]]]] = None,
    ) -> Type[BaseModel]:
        """Return map from function name."""
        spec_functions = spec_functions or self.spec_functions
        if fn_name == "foo":
            return FooSchema
        elif fn_name == "afoo":
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
    assert str(tools[1](True)) == "bar True"
    assert tools[2].metadata.name == "abc"
    assert tools[2].metadata.description == "abc(arg1: str) -> str\n"
    assert tools[2].metadata.fn_schema == AbcSchema

    # test metadata mapping
    tools = tool_spec.to_tool_list(
        func_to_metadata_mapping={
            "foo": ToolMetadata(
                "foo_description", name="foo_name", fn_schema=FooSchema
            ),
        }
    )
    assert len(tools) == 3
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
    assert len(tools) == 3
    assert await tools[0].async_fn("hello", 1) == "foo hello 1"
    assert str(await tools[1].acall(True)) == "bar True"


def test_async_patching() -> None:
    # test sync patching of async function
    tool_spec = TestToolSpec()
    tool_spec.spec_functions = ["afoo"]
    tools = tool_spec.to_tool_list()
    assert len(tools) == 1
    assert tools[0].fn("hello", 1) == "foo hello 1"


def test_tool_spec_schema() -> None:
    """Test tool spec schemas match."""
    tool_spec = TestToolSpec()
    # first is foo, second is bar
    schema1 = tool_spec.get_fn_schema_from_fn_name("foo")
    assert schema1 == FooSchema
    schema2 = tool_spec.get_fn_schema_from_fn_name("bar")
    assert schema2 == BarSchema


def test_tool_spec_subset() -> None:
    """Test tool spec subset."""
    tool_spec = TestToolSpec()
    tools = tool_spec.to_tool_list(spec_functions=["abc"])
    assert len(tools) == 1
    assert tools[0].metadata.name == "abc"
    assert tools[0].metadata.description == "abc(arg1: str) -> str\n"
    assert tools[0].metadata.fn_schema == AbcSchema
