import pytest
import json

from typing import Optional
from llama_index.core.llms import MockLLM
from llama_index.core.llms.mock import MockFunctionCallingLLM, BlockToContentCallback
from llama_index.core.llms.llm import ToolSelection
from llama_index.core.base.llms.types import (
    ChatMessage,
    TextBlock,
    DocumentBlock,
    ImageBlock,
    ToolCallBlock,
    ContentBlock,
)


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
        ToolCallBlock(tool_name="hello", tool_kwargs={}, tool_call_id="4"),
        ToolCallBlock(
            tool_name="divide", tool_kwargs={"x": 1, "y": 0}, tool_call_id="5"
        ),
    ]


@pytest.fixture()
def blocks_to_content_callback() -> BlockToContentCallback:
    def blocks_to_content(
        blocks: list[ContentBlock], tool_calls: Optional[list[ToolCallBlock]] = None
    ) -> str:
        def divide(x: int, y: int) -> int:
            return int(x / y)

        content = ""
        for block in blocks:
            if isinstance(block, TextBlock):
                content += block.text
            elif isinstance(block, ToolCallBlock):
                if block.tool_name == "divide":
                    if isinstance(block.tool_kwargs, dict):
                        try:
                            content += f"<toolcall id={block.tool_call_id}>{divide(**block.tool_kwargs)}</toolcall>"
                        except Exception:
                            content += (
                                f"<toolcall id={block.tool_call_id}>error</toolcall>"
                            )
                    else:
                        try:
                            args = json.loads(block.tool_kwargs)
                            content += f"<toolcall id={block.tool_call_id}>{divide(**args)}</toolcall>"
                        except Exception:
                            content += (
                                f"<toolcall id={block.tool_call_id}>error</toolcall>"
                            )
            else:
                continue
        return content

    return blocks_to_content


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
    llm = MockFunctionCallingLLM()
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


def test_mock_function_calling_llm_tool_calls(
    tool_calls: list[ToolCallBlock],
) -> None:
    llm = MockFunctionCallingLLM(max_tokens=200)
    result = llm.chat(messages=[ChatMessage(role="user", blocks=tool_calls)])
    assert result.message.content == "<empty>"
    assert llm.tool_calls == tool_calls


def test_mock_function_calling_llm_custom_callback(
    tool_calls: list[ToolCallBlock],
    blocks_to_content_callback: BlockToContentCallback,
) -> None:
    llm = MockFunctionCallingLLM(
        max_tokens=200, blocks_to_content_callback=blocks_to_content_callback
    )
    blocks = [TextBlock(text="hello world"), *tool_calls]
    result = llm.chat(messages=[ChatMessage(role="user", blocks=blocks)])
    assert (
        result.message.content
        == "hello world<toolcall id=1>3</toolcall><toolcall id=2>3</toolcall><toolcall id=3>error</toolcall><toolcall id=5>error</toolcall>"
    )


@pytest.mark.asyncio
async def test_mock_function_calling_llm_astream_chat_with_tools(
    messages: list[ChatMessage],
) -> None:
    """Test that astream_chat_with_tools works correctly."""
    llm = MockFunctionCallingLLM(max_tokens=200)

    # Mock tools list (can be empty for this test)
    tools = []

    cont = ""
    stream = await llm.astream_chat_with_tools(tools=tools, chat_history=messages)
    async for s in stream:
        cont += s.message.content or ""

    assert cont == "hello world<document>hello world</document><image>1px</image>"


def test_mock_function_calling_llm_get_tool_calls_from_response() -> None:
    """Test that get_tool_calls_from_response extracts tool calls correctly."""
    llm = MockFunctionCallingLLM(max_tokens=200)

    # Create a response with tool calls in additional_kwargs
    tool_selection = ToolSelection(
        tool_id="test_id",
        tool_name="test_tool",
        tool_kwargs={"arg1": "value1"},
    )

    from llama_index.core.base.llms.types import ChatResponse

    response = ChatResponse(
        message=ChatMessage(
            role="assistant",
            blocks=[
                ToolCallBlock(
                    tool_call_id="test_id",
                    tool_name="test_tool",
                    tool_kwargs={"arg1": "value1"},
                )
            ],
        )
    )

    tool_calls = llm.get_tool_calls_from_response(response)
    assert len(tool_calls) == 1
    assert tool_calls[0].tool_id == tool_selection.tool_id
    assert tool_calls[0].tool_name == tool_selection.tool_name
    assert tool_calls[0].tool_kwargs == tool_selection.tool_kwargs


def test_mock_function_calling_llm_get_tool_calls_from_response_empty() -> None:
    """Test that get_tool_calls_from_response returns empty list when no tool calls."""
    llm = MockFunctionCallingLLM(max_tokens=200)

    from llama_index.core.base.llms.types import ChatResponse

    response = ChatResponse(
        message=ChatMessage(
            role="assistant",
            content="test",
            additional_kwargs={},
        )
    )

    tool_calls = llm.get_tool_calls_from_response(response)
    assert len(tool_calls) == 0
