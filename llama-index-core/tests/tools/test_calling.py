"""Tests for tool invocation helpers."""

import pytest

from llama_index.core.llms.llm import ToolSelection
from llama_index.core.tools.calling import (
    acall_tool,
    acall_tool_with_selection,
    call_tool,
    call_tool_with_selection,
)
from llama_index.core.tools.function_tool import FunctionTool


def test_call_tool_does_not_retry_failed_function_tool() -> None:
    calls = []

    def remote_write(input: str) -> str:
        calls.append(input)
        raise RuntimeError("response lost after request was accepted")

    tool = FunctionTool.from_defaults(remote_write)

    output = call_tool(tool, {"input": "charge"})

    assert output.is_error
    assert calls == ["charge"]


@pytest.mark.asyncio
async def test_acall_tool_does_not_retry_failed_function_tool() -> None:
    calls = []

    async def remote_write(input: str) -> str:
        calls.append(input)
        raise RuntimeError("response lost after request was accepted")

    tool = FunctionTool.from_defaults(async_fn=remote_write)

    output = await acall_tool(tool, {"input": "charge"})

    assert output.is_error
    assert calls == ["charge"]


def test_call_tool_supports_keyword_only_function_tool() -> None:
    def lookup(*, query: str) -> str:
        return query

    tool = FunctionTool.from_defaults(lookup)

    output = call_tool(tool, {"query": "LlamaIndex"})

    assert output.raw_output == "LlamaIndex"


def test_call_tool_supports_positional_only_function_tool() -> None:
    def lookup(query: str, /) -> str:
        return query

    tool = FunctionTool.from_defaults(lookup)

    output = call_tool(tool, {"query": "LlamaIndex"})

    assert output.raw_output == "LlamaIndex"


def test_call_tool_with_selection_unknown_tool() -> None:
    tool = FunctionTool.from_defaults(lambda query: query, name="lookup")
    tool_call = ToolSelection(tool_id="1", tool_name="search", tool_kwargs={"q": "x"})

    output = call_tool_with_selection(tool_call, [tool])

    assert output.is_error
    assert output.tool_name == "search"
    assert output.raw_input == {"q": "x"}


@pytest.mark.asyncio
async def test_acall_tool_with_selection_unknown_tool() -> None:
    tool = FunctionTool.from_defaults(lambda query: query, name="lookup")
    tool_call = ToolSelection(tool_id="1", tool_name="search", tool_kwargs={"q": "x"})

    output = await acall_tool_with_selection(tool_call, [tool])

    assert output.is_error
    assert output.tool_name == "search"
    assert output.raw_input == {"q": "x"}
