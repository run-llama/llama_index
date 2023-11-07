from typing import List
from unittest.mock import MagicMock, patch

from llama_index.llms import OpenAILike
from llama_index.llms.base import ChatMessage
from openai.types import Completion, CompletionChoice
from openai.types.chat.chat_completion import ChatCompletion, Choice
from openai.types.chat.chat_completion_message import ChatCompletionMessage


class MockTokenizer:
    def encode(self, text: str) -> List[str]:
        return text.split(" ")


def test_interfaces() -> None:
    llm = OpenAILike(model="placeholder")
    assert llm.class_name() == type(llm).__name__
    assert llm.model == "placeholder"


def mock_chat_completion(text: str) -> ChatCompletion:
    return ChatCompletion(
        id="chatcmpl-abc123",
        object="chat.completion",
        created=1677858242,
        model="gpt-3.5-turbo-0301",
        usage={"prompt_tokens": 13, "completion_tokens": 7, "total_tokens": 20},
        choices=[
            Choice(
                message=ChatCompletionMessage(role="assistant", content=text),
                finish_reason="stop",
                index=0,
            )
        ],
    )


def mock_completion(text: str) -> Completion:
    return Completion(
        id="chatcmpl-abc123",
        object="text_completion",
        created=1677858242,
        model="gpt-3.5-turbo-0301",
        usage={"prompt_tokens": 13, "completion_tokens": 7, "total_tokens": 20},
        choices=[
            CompletionChoice(
                text=text,
                finish_reason="stop",
                index=0,
            )
        ],
    )


@patch("llama_index.llms.openai.SyncOpenAI")
def test_completion(MockSyncOpenAI: MagicMock) -> None:
    text = "placeholder"

    mock_instance = MockSyncOpenAI.return_value
    mock_instance.completions.create.return_value = mock_completion(text)

    llm = OpenAILike(
        model="placeholder",
        is_chat_model=False,
        context_window=1024,
        tokenizer=MockTokenizer(),
    )

    response = llm.complete("A long time ago in a galaxy far, far away")
    assert response.text == text


@patch("llama_index.llms.openai.SyncOpenAI")
def test_chat(MockSyncOpenAI: MagicMock) -> None:
    content = "placeholder"

    mock_instance = MockSyncOpenAI.return_value
    mock_instance.chat.completions.create.return_value = mock_chat_completion(content)

    llm = OpenAILike(
        model="models/placeholder", is_chat_model=True, tokenizer=MockTokenizer()
    )

    response = llm.chat([ChatMessage(role="user", content="test message")])
    assert response.message.content == content


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
