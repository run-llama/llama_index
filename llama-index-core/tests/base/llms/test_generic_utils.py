"""Tests for the LLM response conversion helpers."""

import asyncio

from llama_index.core.base.llms.generic_utils import (
    astream_chat_response_to_completion_response,
    chat_response_to_completion_response,
    stream_chat_response_to_completion_response,
)
from llama_index.core.base.llms.types import ChatMessage, ChatResponse


def _chat_response(content: str, message_kwargs: dict, response_kwargs: dict):
    message = ChatMessage(
        role="assistant", content=content, additional_kwargs=message_kwargs
    )
    return message, ChatResponse(message=message, additional_kwargs=response_kwargs)


def test_chat_response_conversion_does_not_mutate_message() -> None:
    message, chat_response = _chat_response(
        "hi", {"tool_call_id": "call_1"}, {"usage": {"total_tokens": 3}}
    )

    completion_response = chat_response_to_completion_response(chat_response)

    assert completion_response.text == "hi"
    assert completion_response.additional_kwargs == {
        "tool_call_id": "call_1",
        "usage": {"total_tokens": 3},
    }
    # converting a response must not write the response kwargs into the message
    assert message.additional_kwargs == {"tool_call_id": "call_1"}


def test_stream_chat_conversion_does_not_mutate_message() -> None:
    messages = []
    responses = []
    for content in ("a", "b"):
        message, response = _chat_response(content, {"index": 0}, {"usage": {}})
        messages.append(message)
        responses.append(response)

    completions = list(stream_chat_response_to_completion_response(iter(responses)))

    assert [completion.text for completion in completions] == ["a", "b"]
    assert all(
        completion.additional_kwargs == {"index": 0, "usage": {}}
        for completion in completions
    )
    assert all(message.additional_kwargs == {"index": 0} for message in messages)


def test_astream_chat_conversion_does_not_mutate_message() -> None:
    messages = []
    responses = []
    for content in ("a", "b"):
        message, response = _chat_response(content, {"index": 0}, {"usage": {}})
        messages.append(message)
        responses.append(response)

    async def collect():
        async def gen():
            for response in responses:
                yield response

        return [
            completion
            async for completion in astream_chat_response_to_completion_response(gen())
        ]

    completions = asyncio.run(collect())

    assert [completion.text for completion in completions] == ["a", "b"]
    assert all(
        completion.additional_kwargs == {"index": 0, "usage": {}}
        for completion in completions
    )
    assert all(message.additional_kwargs == {"index": 0} for message in messages)
