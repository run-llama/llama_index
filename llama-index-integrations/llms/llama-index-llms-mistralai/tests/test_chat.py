import asyncio
from unittest.mock import AsyncMock, Mock

import pytest
from mistralai.client import models

from llama_index.core.base.llms.types import ChatMessage, TextBlock, ToolCallBlock
from llama_index.llms.mistralai import MistralAI


@pytest.mark.parametrize("is_async", [False, True])
@pytest.mark.parametrize(
    ("content_fields", "expected_content"),
    [
        ({}, ""),
        ({"content": None}, ""),
        ({"content": ""}, ""),
        ({"content": "Let me check."}, "Let me check."),
    ],
)
@pytest.mark.parametrize("with_tool_call", [False, True])
def test_chat_nullable_content(
    monkeypatch: pytest.MonkeyPatch,
    is_async: bool,
    content_fields: dict[str, str | None],
    expected_content: str,
    with_tool_call: bool,
) -> None:
    llm = MistralAI(api_key="test-key")
    completion = models.ChatCompletionResponse.model_validate(
        {
            "id": "completion-1",
            "object": "chat.completion",
            "created": 1,
            "model": "mistral-large-latest",
            "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
            "choices": [
                {
                    "index": 0,
                    "finish_reason": "tool_calls" if with_tool_call else "stop",
                    "message": {
                        "role": "assistant",
                        **content_fields,
                        "tool_calls": [
                            {
                                "id": "call_1234",
                                "type": "function",
                                "function": {
                                    "name": "weather",
                                    "arguments": '{"city": "Paris"}',
                                },
                            }
                        ]
                        if with_tool_call
                        else None,
                    },
                }
            ],
        }
    )
    messages = [ChatMessage(role="user", content="What is the weather in Paris?")]
    if is_async:
        monkeypatch.setattr(
            llm._client.chat, "complete_async", AsyncMock(return_value=completion)
        )
        response = asyncio.run(llm.achat(messages))
    else:
        monkeypatch.setattr(llm._client.chat, "complete", Mock(return_value=completion))
        response = llm.chat(messages)

    assert response.message.content == expected_content
    assert response.message.blocks == [TextBlock(text=expected_content)] + (
        [
            ToolCallBlock(
                tool_call_id="call_1234",
                tool_name="weather",
                tool_kwargs='{"city": "Paris"}',
            )
        ]
        if with_tool_call
        else []
    )
    calls = llm.get_tool_calls_from_response(response, error_on_no_tool_call=False)
    assert [(call.tool_id, call.tool_name, call.tool_kwargs) for call in calls] == (
        [("call_1234", "weather", {"city": "Paris"})] if with_tool_call else []
    )
