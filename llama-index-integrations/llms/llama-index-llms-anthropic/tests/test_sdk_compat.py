"""SDK compatibility: plain message dicts and sampling kwargs."""

import inspect
from unittest.mock import MagicMock

from llama_index.core.base.llms.types import (
    ChatMessage,
    MessageRole,
    ToolCallBlock,
)
from llama_index.llms.anthropic import Anthropic
from llama_index.llms.anthropic.utils import (
    messages_to_anthropic_beta_messages,
    messages_to_anthropic_messages,
)


def _sample_messages() -> list[ChatMessage]:
    return [
        ChatMessage(role=MessageRole.USER, content="Hello"),
        ChatMessage(
            role=MessageRole.ASSISTANT,
            blocks=[
                ToolCallBlock(
                    tool_name="echo",
                    tool_kwargs={"text": "Hello"},
                    tool_call_id="call_1",
                )
            ],
        ),
        ChatMessage(
            role=MessageRole.TOOL,
            content="Hello",
            additional_kwargs={"tool_call_id": "call_1"},
        ),
    ]


def _assert_plain_message_dicts(messages: list) -> None:
    assert messages
    assert all(type(message) is dict for message in messages)
    assert all({"role", "content"} <= set(message) for message in messages)

    user = messages[0]
    assert user["role"] == "user"
    assert user["content"][0]["type"] == "text"
    assert user["content"][0]["text"] == "Hello"

    assistant = messages[1]
    assert assistant["role"] == "assistant"
    assert assistant["content"][0]["type"] == "tool_use"
    assert assistant["content"][0]["id"] == "call_1"
    assert assistant["content"][0]["name"] == "echo"

    tool_result = messages[2]
    assert tool_result["role"] == "user"
    assert tool_result["content"][0]["type"] == "tool_result"
    assert tool_result["content"][0]["tool_use_id"] == "call_1"
    assert tool_result["content"][0]["content"][0]["text"] == "Hello"


def test_messages_to_anthropic_messages_are_plain_dicts() -> None:
    anthropic_messages, system_prompt = messages_to_anthropic_messages(
        _sample_messages()
    )
    assert system_prompt == ""
    _assert_plain_message_dicts(list(anthropic_messages))


def test_messages_to_anthropic_beta_messages_are_plain_dicts() -> None:
    anthropic_messages, system_prompt = messages_to_anthropic_beta_messages(
        _sample_messages()
    )
    assert system_prompt == ""
    _assert_plain_message_dicts(list(anthropic_messages))


class _FakeMessage(dict):
    """Mapping response whose content blocks can be iterated."""

    def __init__(self) -> None:
        super().__init__()
        self.content: list = []


def _create_signature(*, include_sampling: bool) -> inspect.Signature:
    parameters = [
        inspect.Parameter("max_tokens", inspect.Parameter.KEYWORD_ONLY),
        inspect.Parameter("messages", inspect.Parameter.KEYWORD_ONLY),
        inspect.Parameter("model", inspect.Parameter.KEYWORD_ONLY),
        inspect.Parameter("stream", inspect.Parameter.KEYWORD_ONLY, default=False),
        inspect.Parameter("system", inspect.Parameter.KEYWORD_ONLY, default=None),
    ]
    if include_sampling:
        parameters.extend(
            [
                inspect.Parameter(
                    "temperature", inspect.Parameter.KEYWORD_ONLY, default=None
                ),
                inspect.Parameter(
                    "top_p", inspect.Parameter.KEYWORD_ONLY, default=None
                ),
                inspect.Parameter(
                    "top_k", inspect.Parameter.KEYWORD_ONLY, default=None
                ),
            ]
        )
    return inspect.Signature(parameters)


def _llm_with_create_signature(
    signature: inspect.Signature,
    *,
    model: str = "claude-sonnet-4-5",
    temperature: float = 0.2,
    additional_kwargs: dict | None = None,
) -> Anthropic:
    llm = Anthropic(
        model=model,
        temperature=temperature,
        additional_kwargs=additional_kwargs or {},
        api_key="test-key",
    )
    mock_create = MagicMock(return_value=_FakeMessage())
    mock_create.__signature__ = signature
    mock_client = MagicMock()
    mock_client.messages.create = mock_create
    llm._client = mock_client
    return llm


def test_chat_omits_sampling_kwargs_when_create_rejects_them() -> None:
    llm = _llm_with_create_signature(
        _create_signature(include_sampling=False),
        temperature=0.2,
        additional_kwargs={"top_k": 5},
    )

    llm.chat(
        [ChatMessage(role=MessageRole.USER, content="Hi")],
        top_p=0.5,
        temperature=0.9,
    )

    kwargs = llm._client.messages.create.call_args.kwargs
    assert "temperature" not in kwargs
    assert "top_p" not in kwargs
    assert "top_k" not in kwargs
    assert kwargs["model"] == "claude-sonnet-4-5"
    assert kwargs["max_tokens"] == llm.max_tokens
    # The class field stays available for callers and for SDKs that accept it.
    assert llm.temperature == 0.2


def test_chat_passes_temperature_when_create_accepts_it() -> None:
    llm = _llm_with_create_signature(
        _create_signature(include_sampling=True),
        temperature=0.2,
        additional_kwargs={"top_p": 0.7},
    )

    llm.chat(
        [ChatMessage(role=MessageRole.USER, content="Hi")],
        top_k=20,
    )

    kwargs = llm._client.messages.create.call_args.kwargs
    assert kwargs["temperature"] == 0.2
    assert kwargs["top_p"] == 0.7
    assert kwargs["top_k"] == 20


def test_chat_omits_temperature_for_models_that_forbid_it() -> None:
    llm = _llm_with_create_signature(
        _create_signature(include_sampling=True),
        model="claude-opus-4-7",
        temperature=0.2,
    )

    llm.chat([ChatMessage(role=MessageRole.USER, content="Hi")])

    kwargs = llm._client.messages.create.call_args.kwargs
    assert "temperature" not in kwargs
    assert kwargs["model"] == "claude-opus-4-7"
    assert llm.temperature == 0.2
