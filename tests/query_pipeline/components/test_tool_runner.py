"""Test components."""
from typing import Any, List, Sequence

import pytest
from llama_index.core.query_pipeline.components import (
    ArgPackComponent,
    FnComponent,
    InputComponent,
    KwargPackComponent,
)
from llama_index.prompts.mixin import PromptDictType
from llama_index.query_pipeline.components.router import (
    RouterComponent,
    SelectorComponent,
)
from llama_index.query_pipeline.query import QueryPipeline
from llama_index.schema import QueryBundle
from llama_index.selectors.types import (
    BaseSelector,
    MultiSelection,
    SelectorResult,
    SingleSelection,
)
from llama_index.tools.types import ToolMetadata
from llama_index.tools.function_tool import FunctionTool
from llama_index.query_pipeline.components.tool_runner import ToolRunnerComponent


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
    