from typing import List
from unittest.mock import patch

from llama_index.llms import OpenAILike
from llama_index.llms.base import ChatMessage


class MockTokenizer:
    def encode(self, text: str) -> List[str]:
        return text.split(" ")


def test_interfaces() -> None:
    llm = OpenAILike(model="placeholder")
    assert llm.class_name() == type(llm).__name__
    assert llm.model == "placeholder"


def test_completion() -> None:
    llm = OpenAILike(
        model="placeholder",
        is_chat_model=False,
        context_window=1024,
        tokenizer=MockTokenizer(),
    )

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
        response = llm.complete("A long time ago in a galaxy far, far away")
    assert response.text == text
    mock_completion.assert_called_once()


def test_chat() -> None:
    llm = OpenAILike(
        model="models/placeholder", is_chat_model=True, tokenizer=MockTokenizer()
    )
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


def test_serialization() -> None:
    llm = OpenAILike(
        model="placeholder",
        is_chat_model=True,
        context_window=42,
        tokenizer=MockTokenizer(),
    )

    serialized = llm.to_dict()

    assert serialized["is_chat_model"]
    assert serialized["context_window"] == 42
