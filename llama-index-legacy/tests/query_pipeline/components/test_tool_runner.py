"""Test components."""

from llama_index.legacy.query_pipeline.components.tool_runner import (
    ToolRunnerComponent,
)
from llama_index.legacy.tools.function_tool import FunctionTool
from llama_index.legacy.tools.types import ToolMetadata


def foo_fn(a: int, b: int = 1, c: int = 2) -> int:
    """Foo function."""
    return a + b + c


def test_tool_runner() -> None:
    """Test tool runner."""
    tool_runner_component = ToolRunnerComponent(
        tools=[
            FunctionTool(
                fn=foo_fn,
                metadata=ToolMetadata(
                    name="foo",
                    description="foo",
                ),
            )
        ]
    )

    output = tool_runner_component.run_component(
        tool_name="foo", tool_input={"a": 1, "b": 2, "c": 3}
    )
    assert output["output"].content == "6"
