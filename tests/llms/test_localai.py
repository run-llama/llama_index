from unittest.mock import patch

import pytest
from llama_index.llms import LocalAI
from llama_index.llms.base import ChatMessage


def test_interfaces() -> None:
    llm = LocalAI(model="placeholder")
    assert llm.class_name() == type(llm).__name__
    assert llm.model == "placeholder"


def test_completion() -> None:
    llm = LocalAI(model="models/placeholder.gguf")

    text = "...\n\nIt was just another day at the office. The sun had ris"
    with patch(
        "llama_index.llms.openai.completion_with_retry",
        return_value={
            "id": "123",
            "object": "text_completion",
            "created": 1696036786,
            "model": "models/placeholder.gguf",
            "choices": [
                {
                    "text": text,
                    "index": 0,
                    "logprobs": None,
                    "finish_reason": "length",
                }
            ],
            "usage": {"prompt_tokens": 13, "completion_tokens": 16, "total_tokens": 29},
        },
    ) as mock_completion:
        response = llm.complete(
            "A long time ago in a galaxy far, far away", use_chat_completions=False
        )
    assert response.text == text
    mock_completion.assert_called_once()
    # Check we remove the max_tokens if unspecified
    assert "max_tokens" not in mock_completion.call_args.kwargs


def test_chat() -> None:
    llm = LocalAI(model="models/placeholder.gguf", globally_use_chat_completions=True)
    content = "placeholder"
    with patch(
        "llama_index.llms.openai.completion_with_retry",
        return_value={
            "id": "123",
            "object": "chat.completion",
            "created": 1696283017,
            "model": "models/placeholder.gguf",
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": content,
                    },
                    "finish_reason": "length",
                }
            ],
            "usage": {"prompt_tokens": 5, "completion_tokens": 16, "total_tokens": 21},
        },
    ) as mock_chat:
        response = llm.chat([ChatMessage(role="user", content="test message")])
    assert response.message.content == content
    mock_chat.assert_called_once()
    # Check we remove the max_tokens if unspecified
    assert "max_tokens" not in mock_chat.call_args.kwargs


def test_forgetting_kwarg() -> None:
    llm = LocalAI(model="models/placeholder.gguf")

    with patch(
        "llama_index.llms.openai.completion_with_retry", return_value={}
    ) as mock_completion:
        with pytest.raises(NotImplementedError, match="/chat/completions"):
            llm.complete("A long time ago in a galaxy far, far away")
    mock_completion.assert_not_called()


def test_serialization() -> None:
    llm = LocalAI(model="models/placeholder.gguf", max_tokens=42, context_window=43)

    serialized = llm.to_dict()
    # Check OpenAI base class specifics
    assert serialized["max_tokens"] == 42
    # Check LocalAI subclass specifics
    assert serialized["context_window"] == 43
