from typing import List, Any, Generator
import pytest
from unittest.mock import patch, MagicMock
from llama_index.llms.base import (
    ChatMessage,
    MessageRole,
)
from llama_index.llms.rungpt import RunGptLLM


def mock_completion(*args: Any, **kwargs: Any) -> str:
    # Example taken from rungpt example inferece code on github repo.
    return {
        "id": None,
        "object": "text_completion",
        "created": 1692891018,
        "choices": [
            {"text": "This is an indeed test.", "finish_reason": "length", "index": 0.0}
        ],
        "prompt": "Once upon a time,",
        "usage": {"completion_tokens": 21, "total_tokens": 27, "prompt_tokens": 6},
    }


def mock_chat_completion(*args: Any, **kwargs: Any) -> dict:
    # Example taken from rungpt example inferece code on github repo.
    return {
        "id": None,
        "object": "chat.completion",
        "created": 1692892252,
        "choices": [
            {
                "finish_reason": "length",
                "index": 0.0,
                "message": {"content": "This is an indeed test.", "role": "assistant"},
            }
        ],
        "prompt": "Test prompt",
        "usage": {"completion_tokens": 59, "total_tokens": 103, "prompt_tokens": 44},
    }


def mock_completion_stream(*args: Any, **kwargs: Any) -> Generator[dict, None, None]:
    # Example taken from rungpt example inferece code on github repo.
    events = [
        str(
            {
                "id": None,
                "object": "text_completion",
                "created": 1692891964,
                "choices": [{"text": "This", "finish_reason": None, "index": 0.0}],
                "prompt": "This",
                "usage": {
                    "completion_tokens": 1,
                    "total_tokens": 7,
                    "prompt_tokens": 6,
                },
            }
        ),
        str(
            {
                "id": None,
                "object": "text_completion",
                "created": 1692891964,
                "choices": [{"text": " is", "finish_reason": None, "index": 0.0}],
                "prompt": " is",
                "usage": {
                    "completion_tokens": 2,
                    "total_tokens": 9,
                    "prompt_tokens": 7,
                },
            }
        ),
        str(
            {
                "id": None,
                "object": "text_completion",
                "created": 1692891964,
                "choices": [{"text": " test.", "finish_reason": None, "index": 0.0}],
                "prompt": " test.",
                "usage": {
                    "completion_tokens": 3,
                    "total_tokens": 11,
                    "prompt_tokens": 8,
                },
            }
        ),
    ]
    for event in events:
        yield event


def mock_chat_completion_stream(
    *args: Any, **kwargs: Any
) -> Generator[dict, None, None]:
    # Example taken from rungpt example inferece code on github repo.
    events = [
        str(
            {
                "id": None,
                "object": "chat.completion",
                "created": 1692892378,
                "choices": [
                    {
                        "finish_reason": None,
                        "index": 0.0,
                        "message": {"content": "This", "role": "assistant"},
                    }
                ],
                "prompt": "Mock prompt",
                "usage": {
                    "completion_tokens": 1,
                    "total_tokens": 45,
                    "prompt_tokens": 44,
                },
            }
        ),
        str(
            {
                "id": None,
                "object": "chat.completion",
                "created": 1692892378,
                "choices": [
                    {
                        "finish_reason": None,
                        "index": 0.0,
                        "message": {"content": " is", "role": "assistant"},
                    }
                ],
                "prompt": None,
                "usage": {
                    "completion_tokens": 2,
                    "total_tokens": 47,
                    "prompt_tokens": 45,
                },
            }
        ),
        str(
            {
                "id": None,
                "object": "chat.completion",
                "created": 1692892379,
                "choices": [
                    {
                        "finish_reason": None,
                        "index": 0.0,
                        "message": {"content": " test.", "role": "assistant"},
                    }
                ],
                "prompt": None,
                "usage": {
                    "completion_tokens": 3,
                    "total_tokens": 49,
                    "prompt_tokens": 46,
                },
            }
        ),
    ]
    for event in events:
        yield event


def mock_chat_history(*args: Any, **kwargs: Any) -> List[ChatMessage]:
    return [
        ChatMessage(
            role=MessageRole.USER,
            message="Hello, my name is zihao, major in artificial intelligence.",
        ),
        ChatMessage(
            role=MessageRole.ASSISTANT,
            message="Hello, what can I do for you?",
        ),
        ChatMessage(
            role=MessageRole.USER,
            message="Could you tell me what is my name and major?",
        ),
    ]


def test_init() -> None:
    dummy = RunGptLLM(model="mock model", endpoint="0.0.0.0:51002")
    assert dummy.model == "mock model"
    assert dummy.endpoint == "0.0.0.0:51002"
    assert isinstance(dummy, RunGptLLM)


def test_complete() -> None:
    dummy = RunGptLLM()
    with patch("requests.post") as mock_post:
        mock_post.return_value.json.return_value = mock_completion()
        response = dummy.complete("mock prompt")
        assert response.text == "This is an indeed test."


@pytest.mark.parametrize(
    "chat_history", [mock_chat_history(), tuple(mock_chat_history())]
)
def test_chat(chat_history: List[ChatMessage]) -> None:
    with patch("requests.post") as mock_post:
        mock_post.return_value.json.return_value = mock_chat_completion()
        dummy = RunGptLLM()
        response = dummy.chat(chat_history)
        assert response.message.content == "This is an indeed test."
        assert response.message.role == "assistant"


@pytest.mark.parametrize(
    "chat_history", [mock_chat_history(), tuple(mock_chat_history())]
)
def test_stream_chat(chat_history: List[ChatMessage]) -> None:
    mock_events = [
        MagicMock(data=event_data) for event_data in mock_chat_completion_stream()
    ]
    mock_event_iterator = iter(mock_events)

    with patch("requests.post"), patch("sseclient.SSEClient") as mock_sseclient:
        mock_response = MagicMock()
        mock_response.json.return_value = {}
        type(mock_response).status_code = 200
        mock_sseclient.return_value.events.return_value = mock_event_iterator

        dummy = RunGptLLM()
        response_gen = dummy.stream_chat(chat_history)
        responses = list(response_gen)
        assert responses[-1].message.content == "This is test."
        assert responses[-1].message.role == "assistant"


def test_stream_complete() -> None:
    mock_events = [
        MagicMock(data=event_data) for event_data in mock_completion_stream()
    ]
    mock_event_iterator = iter(mock_events)
    mock_prompt = "A mock prompt"

    with patch("requests.post"), patch("sseclient.SSEClient") as mock_sseclient:
        mock_response = MagicMock()
        mock_response.json.return_value = {}
        type(mock_response).status_code = 200
        mock_sseclient.return_value.events.return_value = mock_event_iterator

        dummy = RunGptLLM()
        response_gen = dummy.stream_complete(mock_prompt)
        responses = list(response_gen)
        assert responses[-1].text == "This is test."
        assert responses[-1].delta == " test."
