from http import HTTPStatus
from types import SimpleNamespace
from typing import AsyncGenerator, List, Sequence
from unittest.mock import patch

import pytest

from llama_index.core.base.llms.types import (
    CompletionResponse,
    ChatMessage,
    ChatResponse,
)
from llama_index.llms.dashscope.base import DashScope
from llama_index.core.base.llms.types import ChatMessage, ChatResponse, MessageRole


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


class DummyMetadata:
    name = "dummy_tool"
    description = "A dummy tool for testing."

    def get_parameters_dict(self):
        return {
            "properties": {
                "param1": {"type": "string", "description": "A test parameter."}
            },
            "required": ["param1"],
            "type": "object",
        }


class DummyTool:
    metadata = DummyMetadata()


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


@pytest.mark.asyncio
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


@pytest.mark.asyncio
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


def test_convert_tool_to_dashscope_format(dashscope_llm):
    """Test _convert_tool_to_dashscope_format correctly converts a tool to the DashScope format."""
    result = dashscope_llm._convert_tool_to_dashscope_format(DummyTool())

    expected = {
        "type": "function",
        "function": {
            "name": "dummy_tool",
            "description": "A dummy tool for testing.",
            "parameters": {
                "type": "object",
                "properties": {
                    "param1": {"type": "string", "description": "A test parameter."}
                },
            },
            "required": ["param1"],
        },
    }

    assert result == expected, f"Expected {expected}, but got {result}"


def test_get_tool_calls_from_response_actual_data(dashscope_llm):
    """Test get_tool_calls_from_response correctly extracts tool calls from a ChatResponse."""
    additional_kwargs = {
        "tool_calls": [
            {
                "index": 0,
                "id": "call_function_id",
                "type": "function",
                "function": {
                    "name": "get_current_weather",
                    "arguments": '{"location":"location"}',
                },
            }
        ]
    }

    chat_message = ChatMessage(
        role=MessageRole.ASSISTANT.value,
        content="",
        additional_kwargs=additional_kwargs,
    )

    chat_response = ChatResponse(message=chat_message, delta="", raw=None)

    tool_selections = dashscope_llm.get_tool_calls_from_response(chat_response)

    assert len(tool_selections) == 1

    selection = tool_selections[0]
    assert selection.tool_id == "call_function_id"
    assert selection.tool_name == "get_current_weather"
    assert selection.tool_kwargs == {"location": "location"}


def test_prepare_chat_with_tools(dashscope_llm):
    """Test _prepare_chat_with_tools correctly prepares chat with tools."""
    tools: Sequence[DummyTool] = [DummyTool()]

    chat_history: List[ChatMessage] = [
        ChatMessage(role="assistant", content="Previous message")
    ]
    user_msg: str = "User's question"

    extra_kwargs = {"extra_param": 123}

    result = dashscope_llm._prepare_chat_with_tools(
        tools=tools,
        user_msg=user_msg,
        chat_history=chat_history,
        verbose=True,
        allow_parallel_tool_calls=False,
        **extra_kwargs,
    )

    assert "messages" in result
    assert "tools" in result
    assert "stream" in result
    assert result["extra_param"] == 123

    messages: List[ChatMessage] = result["messages"]
    assert len(messages) == 2
    assert messages[0].role == "assistant"
    assert messages[0].content == "Previous message"
    assert messages[1].role == "user"
    assert messages[1].content == "User's question"

    tools_spec = result["tools"]
    expected_tool_spec = {
        "type": "function",
        "function": {
            "name": "dummy_tool",
            "description": "A dummy tool for testing.",
            "parameters": {
                "type": "object",
                "properties": {
                    "param1": {"type": "string", "description": "A test parameter."}
                },
            },
            "required": ["param1"],
        },
    }
    assert len(tools_spec) == 1
    assert tools_spec[0] == expected_tool_spec

    assert result["stream"] is True


@pytest.mark.asyncio
async def test_astream_chat_with_tools(monkeypatch, dashscope_llm):
    """
    Test astream_chat method: when the tools parameter is passed,
    astream_call_with_messages should receive this parameter,
    and the additional_kwargs of the returned ChatResponse should contain the correct tool_calls.
    """

    async def fake_async_responses(*args, **kwargs) -> AsyncGenerator:
        expected_tools = [{"dummy": "tool_spec"}]
        assert kwargs.get("tools") == expected_tools, (
            "tools parameter is not passed correctly"
        )

        class FakeOutput:
            def __init__(self) -> None:
                self.choices = [
                    {
                        "message": {
                            "role": "assistant",
                            "content": "Hello, this is a test.",
                            "tool_calls": [
                                {
                                    "index": 0,
                                    "id": "dummy_tool_call_id",
                                    "type": "function",
                                    "function": {
                                        "name": "dummy_tool",
                                        "arguments": '{"param": "value"}',
                                    },
                                }
                            ],
                        }
                    }
                ]

        class FakeResponse:
            def __init__(self) -> None:
                self.status_code = HTTPStatus.OK
                self.output = FakeOutput()

        yield FakeResponse()

    monkeypatch.setattr(
        "llama_index.llms.dashscope.base.astream_call_with_messages",
        fake_async_responses,
    )

    messages = [ChatMessage(role="user", content="Test message")]
    dummy_tools = [{"dummy": "tool_spec"}]

    gen = await dashscope_llm.astream_chat(messages, tools=dummy_tools)
    responses = []
    async for response in gen:
        responses.append(response)

    assert len(responses) == 1
    response = responses[0]
    tool_calls = response.message.additional_kwargs.get("tool_calls", [])
    assert len(tool_calls) == 1, "tool_calls number is not correct"
    expected_tool_call = {
        "index": 0,
        "id": "dummy_tool_call_id",
        "type": "function",
        "function": {"name": "dummy_tool", "arguments": '{"param": "value"}'},
    }
    assert tool_calls[0] == expected_tool_call, "tool_calls is not as expected"
