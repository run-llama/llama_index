from unittest.mock import patch

import pytest

from llama_index.core.base.llms.types import (
    CompletionResponse,
    ChatMessage,
    ChatResponse,
)
from llama_index.llms.dashscope.base import DashScope


@pytest.fixture()
def dashscope_llm():
    return DashScope(api_key="test")


@pytest.fixture()
def dashscope_api_response():
    return {
        "status_code": 200,
        "request_id": "4438deec-2d21-9b9c-b405-a47459fd8f75",
        "code": "",
        "message": "",
        "output": {
            "choices": [
                {
                    "finish_reason": "stop",
                    "message": {"role": "assistant", "content": "hi, there!"},
                }
            ]
        },
        "usage": {"total_tokens": 161, "output_tokens": 91, "input_tokens": 70},
    }


@pytest.fixture()
def prompt() -> str:
    return "hi, there!"


@patch("llama_index.llms.dashscope.base.call_with_messages")
def test_dashscope_complete(
    mock_call_with_messages, dashscope_llm, dashscope_api_response, prompt
):
    mock_call_with_messages.return_value = dashscope_api_response
    response = dashscope_llm.complete(prompt)
    assert isinstance(response, CompletionResponse)
    assert response.text == "hi, there!"


@patch("llama_index.llms.dashscope.base.call_with_messages")
def test_dashscope_chat(
    mock_call_with_messages, dashscope_llm, dashscope_api_response, prompt
):
    mock_call_with_messages.return_value = dashscope_api_response
    response = dashscope_llm.chat(messages=[ChatMessage.from_str(prompt)])
    assert isinstance(response, ChatResponse)
    assert response.message.content == "hi, there!"
