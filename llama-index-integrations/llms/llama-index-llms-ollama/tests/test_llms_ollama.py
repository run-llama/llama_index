import os

import pytest
from ollama import Client
from typing import Annotated

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
    think = response.message.additional_kwargs.get("thinking", None)
    assert think is not None
    assert str(think).strip() != ""


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
    think = response.message.additional_kwargs.get("thinking", None)
    assert think is not None
    assert str(think).strip() != ""


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
    assert response.message.additional_kwargs.get("tool_calls", []) == []
