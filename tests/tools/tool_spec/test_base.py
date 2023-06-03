"""Test tool spec."""

from llama_index.tools.tool_spec.base import BaseToolSpec
from typing import List


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
