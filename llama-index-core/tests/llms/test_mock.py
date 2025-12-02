import pytest
import json

from typing import Callable
from llama_index.core.llms import MockLLM
from llama_index.core.llms.mock import MockFunctionCallingLLM
from llama_index.core.base.llms.types import (
    ChatMessage,
    TextBlock,
    DocumentBlock,
    ImageBlock,
    ToolCallBlock,
)
from llama_index.core.tools.function_tool import FunctionTool


@pytest.fixture()
def messages() -> list[ChatMessage]:
    return [
        ChatMessage(
            role="user",
            blocks=[
                TextBlock(text="hello world"),
                DocumentBlock(data=b"hello world"),
                ImageBlock(image=b"1px"),
            ],
        )
    ]


@pytest.fixture()
def tools() -> dict[str, Callable]:
    def divide(x: float, y: float) -> float:
        """Returns the quotient of two numbers"""
        return x / y

    return {"divide": divide}


@pytest.fixture()
def tool_calls() -> list[ToolCallBlock]:
    return [
        ToolCallBlock(
            tool_name="divide", tool_kwargs={"x": 6, "y": 2}, tool_call_id="1"
        ),
        ToolCallBlock(
            tool_name="divide",
            tool_kwargs=json.dumps({"x": 6, "y": 2}),
            tool_call_id="2",
        ),
        ToolCallBlock(tool_name="divide", tool_kwargs="{", tool_call_id="3"),
        ToolCallBlock(
            tool_name="divide", tool_kwargs={"x": 1, "y": 0}, tool_call_id="4"
        ),
        ToolCallBlock(tool_name="hello", tool_kwargs={}, tool_call_id="5"),
    ]


def test_mock_llm_stream_complete_empty_prompt_no_max_tokens() -> None:
    """
    Test that MockLLM.stream_complete with an empty prompt and max_tokens=None
    does not raise a validation error.
    This test case is based on issue #19353.
    """
    llm = MockLLM(max_tokens=None)
    response_gen = llm.stream_complete("")

    # Consume the generator to trigger the potential error
    responses = list(response_gen)

    # Check that we received a single, empty response
    assert len(responses) == 1
    assert responses[0].text == ""
    assert responses[0].delta == ""


def test_mock_function_calling_llm_init() -> None:
    llm = MockFunctionCallingLLM(max_tokens=200)
    assert llm.max_tokens == 200
    assert llm.metadata.is_function_calling_model


def test_mock_function_calling_llm_sync_methods(messages: list[ChatMessage]) -> None:
    llm = MockFunctionCallingLLM(max_tokens=200)
    result = llm.chat(messages)
    assert (
        result.message.content
        == "hello world<document>hello world</document><image>1px</image>"
    )
    cont = ""
    stream = llm.stream_chat(messages)
    for s in stream:
        cont += s.message.content or ""
    assert cont == "hello world<document>hello world</document><image>1px</image>"


@pytest.mark.asyncio
async def test_mock_function_calling_llm_async_methods(
    messages: list[ChatMessage],
) -> None:
    llm = MockFunctionCallingLLM(max_tokens=200)
    result = await llm.achat(messages)
    assert (
        result.message.content
        == "hello world<document>hello world</document><image>1px</image>"
    )
    cont = ""
    stream = await llm.astream_chat(messages)
    async for s in stream:
        cont += s.message.content or ""
    assert cont == "hello world<document>hello world</document><image>1px</image>"


@pytest.mark.asyncio
async def test_mock_function_calling_llm_tool_calls(
    tools: list[FunctionTool],
    tool_calls: list[ToolCallBlock],
) -> None:
    llm = MockFunctionCallingLLM(max_tokens=200, tools=tools)
    result = await llm.achat(messages=[ChatMessage(role="user", content=tool_calls)])
    assert result.message.content == "<empty>"
    assert llm.tool_calls == tool_calls
    assert len(llm.tool_results) == 4
    assert llm.tool_results.count("error") == 2
    assert llm.tool_results.count(3) == 2
