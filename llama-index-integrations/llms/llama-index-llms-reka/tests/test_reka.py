import pytest
import os
import inspect
from typing import AsyncIterator

from llama_index.llms.reka import RekaLLM
from llama_index.core.base.llms.types import (
    ChatMessage,
    ChatResponse,
    CompletionResponse,
    MessageRole,
)


@pytest.fixture()
def reka_llm():
    api_key = os.getenv("REKA_API_KEY")
    if not api_key:
        pytest.skip("REKA_API_KEY not set in environment variables")
    return RekaLLM(model="reka-core-20240501", api_key=api_key)


def test_chat(reka_llm):
    messages = [
        ChatMessage(role=MessageRole.SYSTEM, content="You are a helpful assistant."),
        ChatMessage(role=MessageRole.USER, content="What is the capital of France?"),
    ]
    response = reka_llm.chat(messages)
    assert isinstance(response, ChatResponse)
    assert response.message.content.strip(), "Chat response should not be empty"
    print(f"\nChat response:\n{response.message.content}")


def test_complete(reka_llm):
    prompt = "The capital of France is"
    response = reka_llm.complete(prompt)
    assert isinstance(response, CompletionResponse)
    assert response.text.strip(), "Completion response should not be empty"
    print(f"\nCompletion response:\n{response.text}")


def test_stream_chat(reka_llm):
    messages = [
        ChatMessage(role=MessageRole.SYSTEM, content="You are a helpful assistant."),
        ChatMessage(
            role=MessageRole.USER,
            content="List the first 5 planets in the solar system.",
        ),
    ]
    stream = reka_llm.stream_chat(messages)
    assert inspect.isgenerator(stream), "stream_chat should return a generator"

    full_response = ""
    for chunk in stream:
        assert isinstance(chunk, ChatResponse)
        full_response += chunk.delta
        print(chunk.delta, end="", flush=True)

    assert full_response.strip(), "Streamed chat response should not be empty"
    print(f"\n\nFull streamed chat response:\n{full_response}")


def test_stream_complete(reka_llm):
    prompt = "List the first 5 planets in the solar system:"
    stream = reka_llm.stream_complete(prompt)
    assert inspect.isgenerator(stream), "stream_complete should return a generator"

    full_response = ""
    for chunk in stream:
        assert isinstance(chunk, CompletionResponse)
        full_response += chunk.delta
        print(chunk.delta, end="", flush=True)

    assert full_response.strip(), "Streamed completion response should not be empty"
    print(f"\n\nFull streamed completion response:\n{full_response}")


@pytest.mark.asyncio()
async def test_achat(reka_llm):
    messages = [
        ChatMessage(role=MessageRole.SYSTEM, content="You are a helpful assistant."),
        ChatMessage(
            role=MessageRole.USER,
            content="What is the largest planet in our solar system?",
        ),
    ]
    response = await reka_llm.achat(messages)
    assert isinstance(response, ChatResponse)
    assert response.message.content.strip(), "Async chat response should not be empty"
    print(f"\nAsync chat response:\n{response.message.content}")


@pytest.mark.asyncio()
async def test_acomplete(reka_llm):
    prompt = "The largest planet in our solar system is"
    response = await reka_llm.acomplete(prompt)
    assert isinstance(response, CompletionResponse)
    assert response.text.strip(), "Async completion response should not be empty"
    print(f"\nAsync completion response:\n{response.text}")


@pytest.mark.asyncio()
async def test_astream_chat(reka_llm):
    messages = [
        ChatMessage(role=MessageRole.SYSTEM, content="You are a helpful assistant."),
        ChatMessage(
            role=MessageRole.USER,
            content="Name the first 5 elements in the periodic table.",
        ),
    ]
    stream = await reka_llm.astream_chat(messages)
    assert isinstance(
        stream, AsyncIterator
    ), "astream_chat should return an async generator"

    full_response = ""
    async for chunk in stream:
        assert isinstance(chunk, ChatResponse)
        full_response += chunk.delta
        print(chunk.delta, end="", flush=True)

    assert full_response.strip(), "Async streamed chat response should not be empty"
    print(f"\n\nFull async streamed chat response:\n{full_response}")


@pytest.mark.asyncio()
async def test_astream_complete(reka_llm):
    prompt = "List the first 5 elements in the periodic table:"
    stream = await reka_llm.astream_complete(prompt)
    assert isinstance(
        stream, AsyncIterator
    ), "astream_complete should return an async generator"

    full_response = ""
    async for chunk in stream:
        assert isinstance(chunk, CompletionResponse)
        full_response += chunk.delta
        print(chunk.delta, end="", flush=True)

    assert (
        full_response.strip()
    ), "Async streamed completion response should not be empty"
    print(f"\n\nFull async streamed completion response:\n{full_response}")
