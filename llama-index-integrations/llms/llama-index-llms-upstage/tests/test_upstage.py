import os
from typing import Any, Generator, Optional
from unittest.mock import MagicMock, patch

from llama_index.core.base.llms.types import ChatMessage

import openai
from openai.types.chat.chat_completion import (
    ChatCompletion,
    ChatCompletionMessage,
    Choice,
)
from openai.types.chat.chat_completion_chunk import ChatCompletionChunk, ChoiceDelta
from openai.types.chat.chat_completion_chunk import Choice as ChunkChoice
from openai.types.completion import CompletionUsage

from llama_index.llms.upstage import Upstage


class CachedUpstageApiKeys:
    """
    Saves the users' Upstage API key either in
    the environment variable or set to the library itself.
    This allows us to run tests by setting it without plowing over
    the local environment.
    """

    def __init__(
        self,
        set_env_key_to: Optional[str] = "",
        set_library_key_to: Optional[str] = None,
        set_fake_key: bool = False,
        set_env_type_to: Optional[str] = "",
        set_library_type_to: str = "upstage",  # default value in upstage package
    ):
        self.set_env_key_to = set_env_key_to
        self.set_library_key_to = set_library_key_to
        self.set_fake_key = set_fake_key
        self.set_env_type_to = set_env_type_to
        self.set_library_type_to = set_library_type_to

    def __enter__(self) -> None:
        self.api_env_variable_was = os.environ.get("UPSTAGE_API_KEY", "")
        self.upstage_api_key_was = openai.api_key

        os.environ["UPSTAGE_API_KEY"] = str(self.set_env_key_to)

        if self.set_fake_key:
            os.environ["UPSTAGE_API_KEY"] = "a" * 32

    # No matter what, set the environment variable back to what it was
    def __exit__(self, *exc: object) -> None:
        os.environ["UPSTAGE_API_KEY"] = str(self.api_env_variable_was)
        openai.api_key = self.upstage_api_key_was


def mock_chat_completion(*args: Any, **kwargs: Any) -> ChatCompletion:
    return ChatCompletion(
        id="chatcmpl-abc123",
        object="chat.completion",
        created=1677858242,
        model="solar-mini",
        usage=CompletionUsage(prompt_tokens=13, completion_tokens=7, total_tokens=20),
        choices=[
            Choice(
                message=ChatCompletionMessage(
                    role="assistant", content="\n\nThis is a test!"
                ),
                finish_reason="stop",
                index=0,
            )
        ],
    )


def mock_chat_completion_stream(
    *args: Any, **kwargs: Any
) -> Generator[ChatCompletionChunk, None, None]:
    responses = [
        ChatCompletionChunk(
            id="chatcmpl-6ptKyqKOGXZT6iQnqiXAH8adNLUzD",
            object="chat.completion.chunk",
            created=1677825464,
            model="solar-mini",
            choices=[
                ChunkChoice(
                    delta=ChoiceDelta(role="assistant"), finish_reason=None, index=0
                )
            ],
        ),
        ChatCompletionChunk(
            id="chatcmpl-6ptKyqKOGXZT6iQnqiXAH8adNLUzD",
            object="chat.completion.chunk",
            created=1677825464,
            model="solar-mini",
            choices=[
                ChunkChoice(
                    delta=ChoiceDelta(content="\n\n"), finish_reason=None, index=0
                )
            ],
        ),
        ChatCompletionChunk(
            id="chatcmpl-6ptKyqKOGXZT6iQnqiXAH8adNLUzD",
            object="chat.completion.chunk",
            created=1677825464,
            model="solar-mini",
            choices=[
                ChunkChoice(delta=ChoiceDelta(content="2"), finish_reason=None, index=0)
            ],
        ),
        ChatCompletionChunk(
            id="chatcmpl-6ptKyqKOGXZT6iQnqiXAH8adNLUzD",
            object="chat.completion.chunk",
            created=1677825464,
            model="solar-mini",
            choices=[ChunkChoice(delta=ChoiceDelta(), finish_reason="stop", index=0)],
        ),
    ]
    yield from responses


@patch("llama_index.llms.openai.base.SyncOpenAI")
def test_chat_model_basic(MockSyncUpstage: MagicMock) -> None:
    with CachedUpstageApiKeys(set_fake_key=True):
        mock_instance = MockSyncUpstage.return_value
        mock_instance.chat.completions.create.return_value = mock_chat_completion()

        llm = Upstage(model="solar-mini")
        prompt = "test prompt"
        message = ChatMessage(role="user", content="test message")

        response = llm.complete(prompt)
        assert response.text == "\n\nThis is a test!"

        chat_response = llm.chat([message])
        assert chat_response.message.content == "\n\nThis is a test!"


@patch("llama_index.llms.openai.base.SyncOpenAI")
def test_chat_model_streaming(MockSyncUpstage: MagicMock) -> None:
    with CachedUpstageApiKeys(set_fake_key=True):
        mock_instance = MockSyncUpstage.return_value
        mock_instance.chat.completions.create.return_value = (
            mock_chat_completion_stream()
        )

        llm = Upstage(model="solar-mini")
        prompt = "test prompt"
        message = ChatMessage(role="user", content="test message")

        response_gen = llm.stream_complete(prompt)
        responses = list(response_gen)
        assert responses[-1].text == "\n\n2"

        mock_instance.chat.completions.create.return_value = (
            mock_chat_completion_stream()
        )
        chat_response_gen = llm.stream_chat([message])
        chat_responses = list(chat_response_gen)
        assert chat_responses[-1].message.content == "\n\n2"
        assert chat_responses[-1].message.role == "assistant"


def test_validates_api_key_is_present() -> None:
    with CachedUpstageApiKeys():
        os.environ["UPSTAGE_API_KEY"] = "a" * 32

        # We can create a new LLM when the env variable is set
        assert Upstage()

        os.environ["UPSTAGE_API_KEY"] = ""

        # We can create a new LLM when the api_key is set on the
        # class directly
        assert Upstage(api_key="a" * 32)
