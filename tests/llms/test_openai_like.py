from typing import List
from unittest.mock import patch

from llama_index.llms import OpenAILike
from llama_index.llms.base import ChatMessage, MessageRole
from llama_index.llms.openai import Tokenizer
from llama_index.llms.openai_like import localai_defaults


class StubTokenizer(Tokenizer):
    def encode(self, text: str) -> List[int]:
        return [sum(ord(letter) for letter in word) for word in text.split(" ")]


STUB_MODEL_NAME = "models/stub.gguf"
STUB_API_KEY = "stub_key"


def test_interfaces() -> None:
    llm = OpenAILike(model=STUB_MODEL_NAME, api_key=STUB_API_KEY)
    assert llm.class_name() == type(llm).__name__
    assert llm.model == STUB_MODEL_NAME


def test_completion() -> None:
    text = "...\n\nIt was just another day at the office. The sun had ris"
    completion_return = {
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
    }

    # NOTE: has no max_tokens or tokenizer, so won't infer max_tokens
    llm = OpenAILike(
        model=STUB_MODEL_NAME,
        **localai_defaults,
        max_tokens=None,
    )
    with patch(
        "llama_index.llms.openai.completion_with_retry", return_value=completion_return
    ) as mock_completion:
        response = llm.complete("A long time ago in a galaxy far, far away")
    assert response.text == text
    mock_completion.assert_called_once_with(
        is_chat_model=False,
        max_retries=10,
        prompt="A long time ago in a galaxy far, far away",
        stream=False,
        api_key="localai_fake",
        api_type="localai_fake",
        api_base="localhost:8080",
        api_version="",
        model=STUB_MODEL_NAME,
        temperature=0.1,
    )

    # NOTE: has tokenizer, so will infer max_tokens
    llm = OpenAILike(
        model=STUB_MODEL_NAME,
        **localai_defaults,
        tokenizer=StubTokenizer(),
    )
    with patch(
        "llama_index.llms.openai.completion_with_retry", return_value=completion_return
    ) as mock_completion:
        response = llm.complete("A long time ago in a galaxy far, far away")
    assert response.text == text
    mock_completion.assert_called_once_with(
        is_chat_model=False,
        max_retries=10,
        prompt="A long time ago in a galaxy far, far away",
        stream=False,
        api_key="localai_fake",
        api_type="localai_fake",
        api_base="localhost:8080",
        api_version="",
        model=STUB_MODEL_NAME,
        temperature=0.1,
        max_tokens=3890,
    )


def test_chat() -> None:
    llm = OpenAILike(model=STUB_MODEL_NAME, **localai_defaults, is_chat_model=True)
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
        response = llm.chat(
            [ChatMessage(role=MessageRole.USER, content="test message")]
        )
    assert response.message.content == content
    mock_chat.assert_called_once_with(
        is_chat_model=True,
        max_retries=10,
        messages=[{"role": MessageRole.USER, "content": "test message"}],
        stream=False,
        api_key="localai_fake",
        api_type="localai_fake",
        api_base="localhost:8080",
        api_version="",
        model=STUB_MODEL_NAME,
        temperature=0.1,
    )


def test_serialization() -> None:
    llm = OpenAILike(
        model=STUB_MODEL_NAME,
        is_chat_model=True,
        api_key=STUB_API_KEY,
        max_tokens=42,
        context_window=43,
        tokenizer=StubTokenizer(),
    )

    serialized = llm.to_dict()
    # Check OpenAI base class specifics
    assert "api_key" not in serialized
    assert serialized["max_tokens"] == 42
    # Check OpenAILike subclass specifics
    assert serialized["context_window"] == 43
    assert serialized["is_chat_model"]
