from unittest.mock import MagicMock, patch

import pytest
from llama_index.llms import LocalAI
from llama_index.llms.base import ChatMessage
from openai.types import Completion, CompletionChoice
from openai.types.chat.chat_completion import ChatCompletion, Choice
from openai.types.chat.chat_completion_message import ChatCompletionMessage


@pytest.mark.filterwarnings("ignore:LocalAI subclass is deprecated")
def test_interfaces() -> None:
    llm = LocalAI(model="placeholder")
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


@pytest.mark.filterwarnings("ignore:LocalAI subclass is deprecated")
@patch("llama_index.llms.openai.SyncOpenAI")
def test_completion(MockSyncOpenAI: MagicMock) -> None:
    text = "placeholder"

    mock_instance = MockSyncOpenAI.return_value
    mock_instance.completions.create.return_value = mock_completion(text)

    llm = LocalAI(model="models/placeholder.gguf")

    response = llm.complete(
        "A long time ago in a galaxy far, far away", use_chat_completions=False
    )
    assert response.text == text


@pytest.mark.filterwarnings("ignore:LocalAI subclass is deprecated")
@patch("llama_index.llms.openai.SyncOpenAI")
def test_chat(MockSyncOpenAI: MagicMock) -> None:
    content = "placeholder"

    mock_instance = MockSyncOpenAI.return_value
    mock_instance.chat.completions.create.return_value = mock_chat_completion(content)

    llm = LocalAI(model="models/placeholder.gguf", globally_use_chat_completions=True)

    response = llm.chat([ChatMessage(role="user", content="test message")])
    assert response.message.content == content


@pytest.mark.filterwarnings("ignore:LocalAI subclass is deprecated")
def test_serialization() -> None:
    llm = LocalAI(model="models/placeholder.gguf", max_tokens=42, context_window=43)

    serialized = llm.to_dict()
    # Check OpenAI base class specifics
    assert serialized["max_tokens"] == 42
    # Check LocalAI subclass specifics
    assert serialized["context_window"] == 43
