from typing import List, Dict, Any, Union, Iterator
from typing_extensions import NotRequired, TypedDict

import pytest
from llama_index.llms.base import (
    ChatMessage,
    ChatResponse,
    MessageRole,
    CompletionResponse,
)
from llama_index.llms.xinference import Xinference

mock_chat_history = [
    ChatMessage(
        role=MessageRole.USER,
        message="mock_chat_history_0",
    ),
    ChatMessage(
        role=MessageRole.ASSISTANT,
        message="mock_chat_history_1",
    ),
    ChatMessage(
        role=MessageRole.USER,
        message="mock_chat_history_2",
    ),
]

mock_chat = {
    "id": "test_id",
    "object": "chat.completion",
    "created": 0,
    "model": "test_model",
    "choices": [
        {
            "index": 0,
            "message": {"role": "assistant", "content": "test_response"},
            "finish_reason": "stop",
        }
    ],
    "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
}

mock_chat_stream = [
    {
        "id": "test_id",
        "model": "test_model",
        "created": 1,
        "object": "chat.completion.chunk",
        "choices": [
            {"index": 0, "delta": {"role": "assistant"}, "finish_reason": None}
        ],
    },
    {
        "id": "test_id",
        "model": "test_model",
        "created": 1,
        "object": "chat.completion.chunk",
        "choices": [
            {
                "index": 0,
                "delta": {"content": "test_response_stream"},
                "finish_reason": None,
            }
        ],
    },
    {
        "id": "test_id",
        "model": "test_model",
        "created": 1,
        "object": "chat.completion.chunk",
        "choices": [{"index": 0, "delta": {"content": " "}, "finish_reason": "length"}],
    },
]


def mock_chat_stream_iterator():
    for i in mock_chat_stream:
        yield i


class MockXinferenceModel:
    def chat(
        self,
        prompt: str,
        chat_history: List[TypedDict],
        generate_config: Dict[str, Any],
    ) -> Union[Iterator[Dict[str, Any]], Dict[str, Any]]:
        assert isinstance(prompt, str)
        if chat_history is not None:
            for chat_item in chat_history:
                assert "role" in chat_item
                assert isinstance(chat_item["role"], str)
                assert "content" in chat_item
                assert isinstance(chat_item["content"], str)

        if "stream" in generate_config and generate_config["stream"] is True:
            return mock_chat_stream_iterator()
        else:
            return mock_chat


class MockRESTfulClient:
    def get_model(self) -> None:
        return MockXinferenceModel()


class MockXinference(Xinference):
    def load(self) -> None:
        try:
            from xinference.client import RESTfulClient
        except ImportError:
            raise ImportError(
                "Could not import Xinference library."
                'Please install Xinference with `pip install "xinference[all]"`'
            )

        self._client = MockRESTfulClient()
        self._generator = self._client.get_model()


def test_init():
    try:
        from xinference.client import RESTfulClient
    except ImportError:
        raise ImportError(
            "Could not import Xinference library."
            'Please install Xinference with `pip install "xinference[all]"`'
        )

    dummy = MockXinference(
        model_uid="uid",
        endpoint="endpoint",
    )
    assert dummy.model_uid == "uid"
    assert dummy.endpoint == "endpoint"
    assert isinstance(dummy._client, MockRESTfulClient)


@pytest.mark.parametrize("chat_history", [mock_chat_history, tuple(mock_chat_history)])
def test_chat(chat_history):
    dummy = MockXinference("uid", "endpoint")
    response = dummy.chat(chat_history)
    assert isinstance(response, ChatResponse)
    assert response.delta is None
    assert response.message.role == MessageRole.ASSISTANT
    assert response.message.content == "test_response"


@pytest.mark.parametrize("chat_history", [mock_chat_history, tuple(mock_chat_history)])
def test_stream_chat(chat_history):
    dummy = MockXinference("uid", "endpoint")
    response_gen = dummy.stream_chat(chat_history)
    total_text = ""
    for i, res in enumerate(response_gen):
        assert i < len(mock_chat_stream)
        assert isinstance(res, ChatResponse)
        assert res.delta == mock_chat_stream[i]["choices"][0]["delta"].get(
            "content", ""
        )
        assert res.message.role == MessageRole.ASSISTANT

        total_text += mock_chat_stream[i]["choices"][0]["delta"].get("content", "")
        assert total_text == res.message.content


def test_complete():
    messages = "test_input"
    dummy = MockXinference("uid", "endpoint")
    response = dummy.complete(messages)
    assert isinstance(response, CompletionResponse)
    assert response.delta is None
    assert response.text == "test_response"


def test_stream_complete():
    message = "test_input"
    dummy = MockXinference("uid", "endpoint")
    response_gen = dummy.stream_complete(prompt=message)
    total_text = ""
    for i, res in enumerate(response_gen):
        assert i < len(mock_chat_stream)
        assert isinstance(res, CompletionResponse)
        assert res.delta == mock_chat_stream[i]["choices"][0]["delta"].get(
            "content", ""
        )

        total_text += mock_chat_stream[i]["choices"][0]["delta"].get("content", "")
        assert total_text == res.text
