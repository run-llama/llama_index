from types import SimpleNamespace
from unittest.mock import patch

import pytest

from llama_index.core.base.llms.types import (
    CompletionResponse,
    ChatMessage,
    ChatResponse,
)
from llama_index.llms.dashscope.base import DashScope


class FakeDashscopeResponse:
    def __init__(self, data: dict):
        self.status_code = data["status_code"]
        self.output = SimpleNamespace(**data["output"])

    def __repr__(self) -> str:
        return f"<FakeDashscopeResponse status_code={self.status_code}>"


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


@pytest.mark.asyncio()
@patch("llama_index.llms.dashscope.base.astream_call_with_messages")
async def test_dashscope_astream_complete(
    mock_astream_call_with_messages, dashscope_llm, dashscope_api_response, prompt
):
    async def async_response_generator():
        yield FakeDashscopeResponse(dashscope_api_response)

    mock_astream_call_with_messages.return_value = async_response_generator()

    responses = []
    gen = await dashscope_llm.astream_complete(prompt)  # 先 await 获取异步生成器
    async for partial_resp in gen:
        responses.append(partial_resp)

    assert len(responses) == 1
    assert isinstance(responses[0], CompletionResponse)
    assert responses[0].text == "hi, there!"
    assert responses[0].delta == "hi, there!"


@pytest.mark.asyncio()
@patch("llama_index.llms.dashscope.base.astream_call_with_messages")
async def test_dashscope_astream_chat(
    mock_astream_call_with_messages, dashscope_llm, dashscope_api_response, prompt
):
    async def async_response_generator():
        yield FakeDashscopeResponse(dashscope_api_response)

    mock_astream_call_with_messages.return_value = async_response_generator()

    responses = []
    gen = await dashscope_llm.astream_chat(messages=[ChatMessage.from_str(prompt)])
    async for partial_chat_resp in gen:
        responses.append(partial_chat_resp)

    assert len(responses) == 1
    assert isinstance(responses[0], ChatResponse)
    assert responses[0].message.content == "hi, there!"
    assert responses[0].delta == "hi, there!"
    assert responses[0].message.role == "assistant"
