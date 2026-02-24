import json
from unittest.mock import MagicMock, patch

import httpx
import pytest

from llama_index.core.base.llms.types import ChatMessage, MessageRole
from llama_index.llms.xortron import Xortron


def test_class_name():
    llm = Xortron(model="xortron-7b")
    assert llm.class_name() == "Xortron_llm"


def test_metadata():
    llm = Xortron(model="xortron-7b", context_window=4096, max_tokens=512)
    metadata = llm.metadata
    assert metadata.model_name == "xortron-7b"
    assert metadata.context_window == 4096
    assert metadata.num_output == 512
    assert metadata.is_chat_model is True


def test_default_init():
    llm = Xortron()
    assert llm.model == "xortron-default"
    assert llm.base_url == "http://localhost:8000"
    assert llm.temperature == 0.7
    assert llm.api_key is None


def test_custom_init():
    llm = Xortron(
        model="xortron-13b",
        base_url="http://custom:9000",
        temperature=0.5,
        api_key="test-key",
        additional_kwargs={"top_p": 0.9},
    )
    assert llm.model == "xortron-13b"
    assert llm.base_url == "http://custom:9000"
    assert llm.temperature == 0.5
    assert llm.api_key == "test-key"
    assert llm.additional_kwargs == {"top_p": 0.9}


def test_get_headers_no_api_key():
    llm = Xortron()
    headers = llm._get_headers()
    assert headers == {"Content-Type": "application/json"}
    assert "Authorization" not in headers


def test_get_headers_with_api_key():
    llm = Xortron(api_key="my-key")
    headers = llm._get_headers()
    assert headers["Authorization"] == "Bearer my-key"


def test_build_payload():
    llm = Xortron(model="xortron-7b", temperature=0.5, max_tokens=256)
    payload = llm._build_payload(prompt="Hello")
    assert payload["model"] == "xortron-7b"
    assert payload["temperature"] == 0.5
    assert payload["max_tokens"] == 256
    assert payload["prompt"] == "Hello"


def test_build_payload_with_messages():
    llm = Xortron(model="xortron-7b")
    messages = [{"role": "user", "content": "Hello"}]
    payload = llm._build_payload(messages=messages)
    assert payload["messages"] == messages
    assert "prompt" not in payload


def test_convert_messages():
    llm = Xortron()
    messages = [
        ChatMessage(role=MessageRole.SYSTEM, content="You are helpful."),
        ChatMessage(role=MessageRole.USER, content="Hello"),
    ]
    converted = llm._convert_messages(messages)
    assert len(converted) == 2
    assert converted[0] == {"role": "system", "content": "You are helpful."}
    assert converted[1] == {"role": "user", "content": "Hello"}


def test_parse_completion_response():
    llm = Xortron()
    data = {"text": "Paris is the capital of France."}
    response = llm._parse_completion_response(data)
    assert response.text == "Paris is the capital of France."
    assert response.raw == data


def test_parse_completion_response_output_key():
    llm = Xortron()
    data = {"output": "Paris is the capital."}
    response = llm._parse_completion_response(data)
    assert response.text == "Paris is the capital."


def test_parse_chat_response():
    llm = Xortron()
    data = {"message": {"role": "assistant", "content": "Hello!"}}
    response = llm._parse_chat_response(data)
    assert response.message.content == "Hello!"
    assert response.raw == data


def test_complete():
    llm = Xortron(model="xortron-7b")
    mock_response = MagicMock()
    mock_response.json.return_value = {"text": "Test response"}
    mock_response.raise_for_status = MagicMock()

    with patch.object(llm, "client", create=True) as mock_client:
        mock_client.post.return_value = mock_response
        response = llm.complete("Test prompt")
        assert response.text == "Test response"
        mock_client.post.assert_called_once()


def test_chat():
    llm = Xortron(model="xortron-7b")
    mock_response = MagicMock()
    mock_response.json.return_value = {
        "message": {"role": "assistant", "content": "Hi there!"}
    }
    mock_response.raise_for_status = MagicMock()

    messages = [ChatMessage(role=MessageRole.USER, content="Hello")]

    with patch.object(llm, "client", create=True) as mock_client:
        mock_client.post.return_value = mock_response
        response = llm.chat(messages)
        assert response.message.content == "Hi there!"
        mock_client.post.assert_called_once()
