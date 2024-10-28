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


mock_response = {
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


def test_dashscope_complete(dashscope_llm, mocker):
    mocker.patch(
        "llama_index.llms.dashscope.base.call_with_messages", return_value=mock_response
    )
    response = dashscope_llm.complete("hi, there!?")
    assert isinstance(response, CompletionResponse)
    assert response.text == "hi, there!"


def test_dashscope_chat(dashscope_llm, mocker):
    mocker.patch(
        "llama_index.llms.dashscope.base.call_with_messages", return_value=mock_response
    )
    response = dashscope_llm.chat(messages=[ChatMessage.from_str("hi, there!?")])
    assert isinstance(response, ChatResponse)
    assert response.message.content == "hi, there!"


@pytest.mark.asyncio()
async def test_dashscope_acomplete(dashscope_llm, mocker):
    mocker.patch(
        "llama_index.llms.dashscope.base.acall_with_messages",
        return_value=mock_response,
    )
    response = await dashscope_llm.acomplete("hi, there!?")
    assert isinstance(response, CompletionResponse)
    assert response.text == "hi, there!"


@pytest.mark.asyncio()
async def test_dashscope_achat(dashscope_llm, mocker):
    mocker.patch(
        "llama_index.llms.dashscope.base.acall_with_messages",
        return_value=mock_response,
    )
    response = await dashscope_llm.achat(messages=[ChatMessage.from_str("hi, there!?")])
    assert isinstance(response, ChatResponse)
    assert response.message.content == "hi, there!"
