"""Tests for tool invocation helpers."""

import pytest

from llama_index.core.tools.calling import acall_tool, call_tool
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
