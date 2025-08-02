import pytest
from pytest_httpx import HTTPXMock

from llama_index.llms.heroku import Heroku
from llama_index.core.llms import ChatMessage, MessageRole


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
