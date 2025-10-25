import os

import pytest
from ollama import Client
from typing import Annotated

from llama_index.core.base.llms.types import ThinkingBlock, TextBlock, ToolCallBlock
from llama_index.core.base.llms.base import BaseLLM
from llama_index.core.bridge.pydantic import BaseModel, Field
from llama_index.core.llms import ChatMessage
from llama_index.core.tools import FunctionTool
from llama_index.llms.ollama import Ollama

test_model = os.environ.get("OLLAMA_TEST_MODEL", "llama3.1:latest")
thinking_test_model = os.environ.get("OLLAMA_THINKING_TEST_MODEL", "qwen3:0.6b")

try:
    client = Client()
    models = client.list()

    model_found = False
    for model in models["models"]:
        if model.model == test_model:
            model_found = True
            break

    if not model_found:
        client = None  # type: ignore
except Exception:
    client = None  # type: ignore


class Song(BaseModel):
    """A song with name and artist."""

    artist_name: str = Field(description="The name of the artist")
    song_name: str = Field(description="The name of the song")


def generate_song(
    artist_name: Annotated[str, "The name of the artist"],
    song_name: Annotated[str, "The name of the song"],
) -> Song:
    """Generates a song with provided name and artist."""
    return Song(artist_name=artist_name, song_name=song_name)


tool = FunctionTool.from_defaults(fn=generate_song)


def test_embedding_class() -> None:
    names_of_base_classes = [b.__name__ for b in Ollama.__mro__]
    assert BaseLLM.__name__ in names_of_base_classes


@pytest.mark.skipif(
    client is None, reason="Ollama client is not available or test model is missing"
)
def test_ollama_chat() -> None:
    llm = Ollama(model=test_model)
    response = llm.chat([ChatMessage(role="user", content="Hello!")])
    assert response is not None
    assert str(response).strip() != ""


@pytest.mark.skipif(
    client is None, reason="Ollama client is not available or test model is missing"
)
def test_ollama_complete() -> None:
    llm = Ollama(model=test_model)
    response = llm.complete("Hello!")
    assert response is not None
    assert str(response).strip() != ""


@pytest.mark.skipif(
    client is None, reason="Ollama client is not available or test model is missing"
)
def test_ollama_stream_chat() -> None:
    llm = Ollama(model=test_model)
    response = llm.stream_chat([ChatMessage(role="user", content="Hello!")])
    for r in response:
        assert r is not None
        assert r.delta is not None
        assert str(r).strip() != ""


@pytest.mark.skipif(
    client is None, reason="Ollama client is not available or test model is missing"
)
def test_ollama_stream_complete() -> None:
    llm = Ollama(model=test_model)
    response = llm.stream_complete("Hello!")
    for r in response:
        assert r is not None
        assert r.delta is not None
        assert str(r).strip() != ""


@pytest.mark.skipif(
    client is None, reason="Ollama client is not available or test model is missing"
)
@pytest.mark.asyncio
async def test_ollama_async_chat() -> None:
    llm = Ollama(model=test_model)
    response = await llm.achat([ChatMessage(role="user", content="Hello!")])
    assert response is not None
    assert str(response).strip() != ""


@pytest.mark.skipif(
    client is None, reason="Ollama client is not available or test model is missing"
)
@pytest.mark.asyncio
async def test_ollama_async_complete() -> None:
    llm = Ollama(model=test_model)
    response = await llm.acomplete("Hello!")
    assert response is not None
    assert str(response).strip() != ""


@pytest.mark.skipif(
    client is None, reason="Ollama client is not available or test model is missing"
)
@pytest.mark.asyncio
async def test_ollama_async_stream_chat() -> None:
    llm = Ollama(model=test_model)
    response = await llm.astream_chat([ChatMessage(role="user", content="Hello!")])
    async for r in response:
        assert r is not None
        assert r.delta is not None
        assert str(r).strip() != ""


@pytest.mark.skipif(
    client is None, reason="Ollama client is not available or test model is missing"
)
@pytest.mark.asyncio
async def test_ollama_async_stream_complete() -> None:
    llm = Ollama(model=test_model)
    response = await llm.astream_complete("Hello!")
    async for r in response:
        assert r is not None
        assert r.delta is not None
        assert str(r).strip() != ""


@pytest.mark.skipif(
    client is None, reason="Ollama client is not available or test model is missing"
)
def test_chat_with_tools() -> None:
    llm = Ollama(model=test_model, context_window=8000)
    response = llm.chat_with_tools(
        [tool], user_msg="Hello! Generate a random artist and song."
    )
    tool_calls = llm.get_tool_calls_from_response(response)
    assert len(tool_calls) == 1
    assert tool_calls[0].tool_name == tool.metadata.name

    tool_result = tool(**tool_calls[0].tool_kwargs)
    assert tool_result.raw_output is not None
    assert isinstance(tool_result.raw_output, Song)


@pytest.mark.skipif(
    client is None, reason="Ollama client is not available or test model is missing"
)
def test_stream_chat_with_tools() -> None:
    """Makes sure that stream chat with tools returns tool call message without any errors"""
    llm = Ollama(model=test_model, context_window=8000)
    response = llm.stream_chat_with_tools(
        [tool], user_msg="Hello! Generate a random artist and song."
    )

    for r in response:
        tool_calls = llm.get_tool_calls_from_response(r)
        assert len(tool_calls) == 1
        assert tool_calls[0].tool_name == tool.metadata.name

        tool_result = tool(**tool_calls[0].tool_kwargs)
        assert tool_result.raw_output is not None
        assert isinstance(tool_result.raw_output, Song)


@pytest.mark.skipif(
    client is None, reason="Ollama client is not available or test model is missing"
)
@pytest.mark.asyncio
async def test_async_chat_with_tools() -> None:
    llm = Ollama(model=test_model, context_window=8000)
    response = await llm.achat_with_tools(
        [tool], user_msg="Hello! Generate a random artist and song."
    )
    tool_calls = llm.get_tool_calls_from_response(response)
    assert len(tool_calls) == 1
    assert tool_calls[0].tool_name == tool.metadata.name

    tool_result = tool(**tool_calls[0].tool_kwargs)
    assert tool_result.raw_output is not None
    assert isinstance(tool_result.raw_output, Song)


@pytest.mark.skipif(
    client is None, reason="Ollama client is not available or test model is missing"
)
def test_chat_with_think() -> None:
    llm = Ollama(model=thinking_test_model, thinking=True, request_timeout=360)
    response = llm.chat(
        [ChatMessage(role="user", content="Hello! What is 32 * 4?")], think=False
    )
    assert response is not None
    assert str(response).strip() != ""
    assert (
        len(
            [
                block
                for block in response.message.blocks
                if isinstance(block, ThinkingBlock)
            ]
        )
        > 0
    )
    assert (
        "".join(
            [
                block.content or ""
                for block in response.message.blocks
                if isinstance(block, ThinkingBlock)
            ]
        )
        != ""
    )


@pytest.mark.skipif(
    client is None, reason="Ollama client is not available or test model is missing"
)
def test_chat_with_thinking_input() -> None:
    llm = Ollama(model=thinking_test_model, thinking=True, request_timeout=360)
    response = llm.chat(
        [
            ChatMessage(role="user", content="Hello! What is 32 * 4?"),
            ChatMessage(
                role="assistant",
                blocks=[
                    ThinkingBlock(
                        content="The user is asking me to multiply two numbers, so I should reply concisely"
                    ),
                    TextBlock(text="128"),
                ],
            ),
            ChatMessage(
                role="user",
                content="Based on your previous reasoning, can you now tell me the result of 50*200?",
            ),
        ],
        think=False,
    )
    assert response is not None
    assert str(response).strip() != ""
    assert (
        len(
            [
                block
                for block in response.message.blocks
                if isinstance(block, ThinkingBlock)
            ]
        )
        > 0
    )
    assert (
        "".join(
            [
                block.content or ""
                for block in response.message.blocks
                if isinstance(block, ThinkingBlock)
            ]
        )
        != ""
    )


@pytest.mark.skipif(
    client is None, reason="Ollama client is not available or test model is missing"
)
@pytest.mark.asyncio
async def test_async_chat_with_think() -> None:
    llm = Ollama(model=thinking_test_model, thinking=True)
    response = await llm.achat(
        [ChatMessage(role="user", content="Hello! What is 32 * 4?")], think=False
    )
    assert response is not None
    assert str(response).strip() != ""
    assert (
        len(
            [
                block
                for block in response.message.blocks
                if isinstance(block, ThinkingBlock)
            ]
        )
        > 0
    )
    assert (
        "".join(
            [
                block.content or ""
                for block in response.message.blocks
                if isinstance(block, ThinkingBlock)
            ]
        )
        != ""
    )


@pytest.mark.skipif(
    client is None, reason="Ollama client is not available or test model is missing"
)
def test_chat_with_tools_returns_empty_array_if_no_tools_were_called() -> None:
    """Make sure get_tool_calls_from_response can gracefully handle no tools in response"""
    llm = Ollama(model=test_model, context_window=1000)
    response = llm.chat(
        tools=[],
        messages=[
            ChatMessage(
                role="system",
                content="You are a useful tool calling agent.",
            ),
            ChatMessage(role="user", content="Hello, how are you?"),
        ],
    )

    assert response.message.additional_kwargs.get("tool_calls", []) == []

    tool_calls = llm.get_tool_calls_from_response(response, error_on_no_tool_call=False)
    assert len(tool_calls) == 0


@pytest.mark.skipif(
    client is None, reason="Ollama client is not available or test model is missing"
)
@pytest.mark.asyncio
async def test_async_chat_with_tools_returns_empty_array_if_no_tools_were_called() -> (
    None
):
    """
    Test that achat returns [] for no tool calls since subsequent processes expect []
    instead of None
    """
    llm = Ollama(model=test_model, context_window=1000)
    response = await llm.achat(
        tools=[],
        messages=[
            ChatMessage(
                role="system",
                content="You are a useful tool calling agent.",
            ),
            ChatMessage(role="user", content="Hello, how are you?"),
        ],
    )
    assert (
        len(
            [
                block
                for block in response.message.blocks
                if isinstance(block, ToolCallBlock)
            ]
        )
        == 0
    )


@pytest.mark.skipif(
    client is None, reason="Ollama client is not available or test model is missing"
)
@pytest.mark.asyncio
async def test_chat_methods_with_tool_input() -> None:
    llm = Ollama(model=thinking_test_model)
    input_messages = [
        ChatMessage(
            role="user",
            content="Hello, can you tell me what is the weather today in London?",
        ),
        ChatMessage(
            role="assistant",
            blocks=[
                ThinkingBlock(
                    content="The user is asking for the weather in London, so I should use the get_weather tool"
                ),
                ToolCallBlock(
                    tool_name="get_weather_tool", tool_kwargs={"location": "London"}
                ),
                TextBlock(
                    text="The weather in London is rainy with a temperature of 15°C."
                ),
            ],
        ),
        ChatMessage(
            role="user",
            content="Can you tell me what input did you give to the 'get_weather' tool? (do not call any other tool)",
        ),
    ]
    response = llm.chat(messages=input_messages)
    assert response.message.content is not None
    assert (
        len(
            [
                block
                for block in response.message.blocks
                if isinstance(block, ToolCallBlock)
            ]
        )
        == 0
    )
    aresponse = await llm.achat(messages=input_messages)
    assert aresponse.message.content is not None
    assert (
        len(
            [
                block
                for block in aresponse.message.blocks
                if isinstance(block, ToolCallBlock)
            ]
        )
        == 0
    )
    response_stream = llm.stream_chat(messages=input_messages)
    blocks = []
    for r in response_stream:
        blocks.extend(r.message.blocks)
    assert len([block for block in blocks if isinstance(block, TextBlock)]) > 0
    assert len([block for block in blocks if isinstance(block, ToolCallBlock)]) == 0
    aresponse_stream = await llm.astream_chat(messages=input_messages)
    ablocks = []
    async for r in aresponse_stream:
        ablocks.extend(r.message.blocks)
    assert len([block for block in ablocks if isinstance(block, TextBlock)]) > 0
    assert len([block for block in ablocks if isinstance(block, ToolCallBlock)]) == 0
