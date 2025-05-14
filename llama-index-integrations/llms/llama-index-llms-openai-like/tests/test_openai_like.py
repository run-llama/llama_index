from types import MappingProxyType
from typing import Any, Dict, List
from unittest.mock import MagicMock, call, patch

from llama_index.core.base.llms.types import ChatMessage, MessageRole
from llama_index.llms.openai import Tokenizer
from llama_index.llms.openai_like import OpenAILike
from openai.types import Completion, CompletionChoice
from openai.types.chat.chat_completion import (
    ChatCompletion,
    Choice,
)
from openai.types.chat.chat_completion_message import (
    ChatCompletionMessage,
)


class StubTokenizer(Tokenizer):
    def encode(self, text: str) -> List[int]:
        return [sum(ord(letter) for letter in word) for word in text.split(" ")]


STUB_MODEL_NAME = "models/stub.gguf"
STUB_API_KEY = "stub_key"

# Use these as kwargs for OpenAILike to connect to LocalAIs
DEFAULT_LOCALAI_PORT = 8080
# TODO: move to MappingProxyType[str, Any] once Python 3.9+
LOCALAI_DEFAULTS: Dict[str, Any] = MappingProxyType(  # type: ignore[assignment]
    {
        "api_key": "localai_fake",
        "api_type": "localai_fake",
        "api_base": f"http://localhost:{DEFAULT_LOCALAI_PORT}/v1",
    }
)


def test_interfaces() -> None:
    llm = OpenAILike(model=STUB_MODEL_NAME, api_key=STUB_API_KEY)
    assert llm.class_name() == type(llm).__name__
    assert llm.model == STUB_MODEL_NAME


def mock_chat_completion(text: str) -> ChatCompletion:
    return ChatCompletion(
        id="chatcmpl-abc123",
        object="chat.completion",
        created=1677858242,
        model=STUB_MODEL_NAME,
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
        id="cmpl-abc123",
        object="text_completion",
        created=1677858242,
        model=STUB_MODEL_NAME,
        usage={"prompt_tokens": 13, "completion_tokens": 7, "total_tokens": 20},
        choices=[
            CompletionChoice(
                text=text,
                finish_reason="stop",
                index=0,
            )
        ],
    )


@patch("llama_index.llms.openai.base.SyncOpenAI")
def test_completion(MockSyncOpenAI: MagicMock) -> None:
    mock_instance = MockSyncOpenAI.return_value
    mock_instance.completions.create.side_effect = [
        mock_completion("1"),
        mock_completion("2"),
    ]

    llm = OpenAILike(
        **LOCALAI_DEFAULTS, model=STUB_MODEL_NAME, context_window=1024, max_tokens=None
    )
    response = llm.complete("A long time ago in a galaxy far, far away")
    expected_calls = [
        # NOTE: has no max_tokens or tokenizer, so won't infer max_tokens
        call(
            prompt="A long time ago in a galaxy far, far away",
            stream=False,
            model=STUB_MODEL_NAME,
            temperature=0.1,
        )
    ]
    assert response.text == "1"
    mock_instance.completions.create.assert_has_calls(expected_calls)

    llm = OpenAILike(
        model=STUB_MODEL_NAME,
        context_window=1024,
        tokenizer=StubTokenizer(),
    )
    response = llm.complete("A long time ago in a galaxy far, far away")
    expected_calls += [
        # NOTE: has tokenizer, so will infer max_tokens
        call(
            prompt="A long time ago in a galaxy far, far away",
            stream=False,
            model=STUB_MODEL_NAME,
            temperature=0.1,
            max_tokens=1014,
        )
    ]
    assert response.text == "2"
    mock_instance.completions.create.assert_has_calls(expected_calls)


@patch("llama_index.llms.openai.base.SyncOpenAI")
def test_chat(MockSyncOpenAI: MagicMock) -> None:
    content = "placeholder"

    mock_instance = MockSyncOpenAI.return_value
    mock_instance.chat.completions.create.return_value = mock_chat_completion(content)

    llm = OpenAILike(
        model=STUB_MODEL_NAME,
        is_chat_model=True,
        tokenizer=StubTokenizer(),
    )

    response = llm.chat([ChatMessage(role=MessageRole.USER, content="test message")])
    assert response.message.content == content
    mock_instance.chat.completions.create.assert_called_once_with(
        messages=[{"role": "user", "content": "test message"}],
        stream=False,
        model=STUB_MODEL_NAME,
        temperature=0.1,
    )

    llm = OpenAILike(
        model=STUB_MODEL_NAME,
        is_chat_model=True,
        tokenizer=StubTokenizer(),
    )

    response = llm.chat([ChatMessage(role=MessageRole.USER, content="test message")])
    assert response.message.content == content
    mock_instance.chat.completions.create.assert_called_with(
        messages=[{"role": "user", "content": "test message"}],
        stream=False,
        model=STUB_MODEL_NAME,
        temperature=0.1,
    )


def test_serialization() -> None:
    llm = OpenAILike(
        model=STUB_MODEL_NAME,
        is_chat_model=True,
        max_tokens=42,
        context_window=43,
        tokenizer=StubTokenizer(),
    )

    serialized = llm.to_dict()
    # Check OpenAI base class specifics
    assert serialized["max_tokens"] == 42
    # Check OpenAILike subclass specifics
    assert serialized["context_window"] == 43
    assert serialized["is_chat_model"]
