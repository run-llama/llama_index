from typing import Annotated
import pytest
from pytest_httpx import HTTPXMock

from llama_index.llms.heroku import Heroku
from llama_index.core.llms import ChatMessage, MessageRole
from llama_index.core.tools import FunctionTool


@pytest.fixture()
def mock_heroku_chat_completion(httpx_mock: HTTPXMock):
    """Mock Heroku chat completion endpoint response."""
    mock_response = {
        "id": "chatcmpl-123",
        "object": "chat.completion",
        "created": 1677652288,
        "model": "claude-3-5-haiku",
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": "Hello! I'm here to help you with any questions you might have.",
                },
                "finish_reason": "stop",
            }
        ],
        "usage": {"prompt_tokens": 9, "completion_tokens": 12, "total_tokens": 21},
    }

    httpx_mock.add_response(
        url="https://test-app.herokuapp.com/v1/chat/completions",
        method="POST",
        json=mock_response,
        status_code=200,
        match_headers={"Authorization": "Bearer test-key"},
    )


@pytest.fixture()
def mock_heroku_completion(httpx_mock: HTTPXMock):
    """Mock Heroku completion endpoint response."""
    mock_response = {
        "id": "cmpl-123",
        "object": "text_completion",
        "created": 1677652288,
        "model": "claude-3-5-haiku",
        "choices": [
            {
                "text": "This is a test completion response.",
                "index": 0,
                "logprobs": None,
                "finish_reason": "stop",
            }
        ],
        "usage": {"prompt_tokens": 5, "completion_tokens": 8, "total_tokens": 13},
    }

    httpx_mock.add_response(
        url="https://test-app.herokuapp.com/v1/completions",
        method="POST",
        json=mock_response,
        status_code=200,
        match_headers={"Authorization": "Bearer test-key"},
    )


@pytest.fixture()
def mock_heroku_tool_call_completion(httpx_mock: HTTPXMock):
    """Mock Heroku tool call completion endpoint response."""
    mock_response = {
        "id": "chatcmpl-1839adcc2079997417288",
        "object": "chat.completion",
        "created": 1745617422,
        "model": "claude-4-sonnet",
        "system_fingerprint": "heroku-inf-1y38gdr",
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": "I'll help you check the current weather in Portland. Since Portland could refer to either Portland, Oregon or Portland, Maine, I should specify the state.\nI'll check Portland, OR as it's the larger and more commonly referenced Portland.",
                    "refusal": None,
                    "tool_calls": [
                        {
                            "id": "tooluse_aFByQsacQ_2BmYMGHvkBmg",
                            "type": "function",
                            "function": {
                                "name": "get_current_weather",
                                "arguments": '{"location":"Portland, OR"}',
                            },
                        }
                    ],
                },
                "finish_reason": "tool_calls",
            }
        ],
        "usage": {"prompt_tokens": 407, "completion_tokens": 107, "total_tokens": 514},
    }

    # More flexible mock that matches any POST request to the chat completions endpoint
    httpx_mock.add_response(
        url="https://test-app.herokuapp.com/v1/chat/completions",
        method="POST",
        json=mock_response,
        status_code=200,
        match_headers={"Authorization": "Bearer test-key"},
    )


@pytest.mark.usefixtures("mock_heroku_chat_completion")
def test_chat_completion() -> None:
    """Test chat completion functionality."""
    llm = Heroku(
        model="claude-3-5-haiku",
        api_key="test-key",
        inference_url="https://test-app.herokuapp.com",
        is_chat_model=True,
    )

    messages = [ChatMessage(role=MessageRole.USER, content="Hello, how are you?")]

    response = llm.chat(messages)
    assert (
        response.message.content
        == "Hello! I'm here to help you with any questions you might have."
    )


@pytest.mark.usefixtures("mock_heroku_completion")
def test_text_completion() -> None:
    """Test text completion functionality."""
    llm = Heroku(
        model="claude-3-5-haiku",
        api_key="test-key",
        inference_url="https://test-app.herokuapp.com",
        is_chat_model=False,
    )

    response = llm.complete("Test prompt")
    assert response.text == "This is a test completion response."


@pytest.mark.usefixtures("mock_heroku_chat_completion")
def test_chat_with_system_message() -> None:
    """Test chat with system message."""
    llm = Heroku(
        model="claude-3-5-haiku",
        api_key="test-key",
        inference_url="https://test-app.herokuapp.com",
        is_chat_model=True,
    )

    messages = [
        ChatMessage(role=MessageRole.SYSTEM, content="You are a helpful assistant."),
        ChatMessage(role=MessageRole.USER, content="Hello, how are you?"),
    ]

    response = llm.chat(messages)
    assert (
        response.message.content
        == "Hello! I'm here to help you with any questions you might have."
    )


@pytest.mark.usefixtures("mock_heroku_chat_completion")
def test_chat_with_max_tokens() -> None:
    """Test chat with max_tokens parameter."""
    llm = Heroku(
        model="claude-3-5-haiku",
        api_key="test-key",
        inference_url="https://test-app.herokuapp.com",
        is_chat_model=True,
        max_tokens=50,
    )

    messages = [ChatMessage(role=MessageRole.USER, content="Hello, how are you?")]

    response = llm.chat(messages)
    assert (
        response.message.content
        == "Hello! I'm here to help you with any questions you might have."
    )


def test_class_name() -> None:
    """Test that class_name returns correct value."""
    llm = Heroku(
        model="claude-3-5-haiku",
        api_key="test-key",
        inference_url="https://test-app.herokuapp.com",
        is_chat_model=True,
    )
    assert llm.class_name() == "Heroku"


@pytest.mark.usefixtures("mock_heroku_tool_call_completion")
def test_chat_with_tool_call_completion() -> None:
    """Test chat with tool call completion."""
    llm = Heroku(
        model="claude-4-sonnet",
        api_key="test-key",
        inference_url="https://test-app.herokuapp.com",
        is_chat_model=True,
    )

    weather_tool = FunctionTool.from_defaults(get_current_weather)

    # Test direct tool calling with the LLM
    messages = [
        ChatMessage(
            role=MessageRole.USER,
            content="What is the weather in Portland?",
            tools=[weather_tool],
        )
    ]

    response = llm.chat(messages)

    # Verify the response contains tool calls
    assert response.message.additional_kwargs.get("tool_calls") is not None

    tool_calls = response.message.additional_kwargs["tool_calls"]
    assert len(tool_calls) > 0
    assert tool_calls[0].function.name == "get_current_weather"


def get_current_weather(
    location: Annotated[str, "A city name and state, formatted like '<name>, <state>'"],
) -> str:
    """Get the current weather in a given location."""
    return f"The current weather in {location} is sunny."
