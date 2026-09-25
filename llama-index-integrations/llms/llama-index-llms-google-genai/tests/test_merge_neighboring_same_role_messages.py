import pytest

from llama_index.core.llms import ChatMessage, MessageRole
from llama_index.llms.google_genai.utils import (
    merge_neighboring_same_role_messages,
    prepare_chat_params,
)


def _texts(message: ChatMessage) -> list[str]:
    return [block.text for block in message.blocks if hasattr(block, "text")]


def test_merge_does_not_mutate_original_messages() -> None:
    """The merge operates on a copy, so the caller's messages stay intact."""
    messages = [
        ChatMessage(role=MessageRole.USER, content="question one"),
        ChatMessage(role=MessageRole.USER, content="question two"),
        ChatMessage(role=MessageRole.ASSISTANT, content="answer"),
    ]

    merged = merge_neighboring_same_role_messages(messages)

    assert _texts(messages[0]) == ["question one"]
    assert _texts(messages[1]) == ["question two"]
    assert _texts(merged[0]) == ["question one", "question two"]
    assert _texts(merged[1]) == ["answer"]


def test_merge_does_not_mutate_original_additional_kwargs() -> None:
    messages = [
        ChatMessage(role=MessageRole.USER, content="first", additional_kwargs={"a": 1}),
        ChatMessage(
            role=MessageRole.USER, content="second", additional_kwargs={"b": 2}
        ),
    ]

    merged = merge_neighboring_same_role_messages(messages)

    assert messages[0].additional_kwargs == {"a": 1}
    assert messages[1].additional_kwargs == {"b": 2}
    assert merged[0].additional_kwargs == {"a": 1, "b": 2}


@pytest.mark.asyncio
async def test_prepare_chat_params_does_not_mutate_caller_history() -> None:
    """
    Chat memory reuses the same ChatMessage objects on every turn, so
    mutating them corrupts the request and compounds with each chat() call.
    """
    history = [
        ChatMessage(role=MessageRole.USER, content="question one"),
        ChatMessage(role=MessageRole.USER, content="question two"),
    ]

    await prepare_chat_params("gemini-2.5-flash", list(history), "inline", None)
    assert _texts(history[0]) == ["question one"]

    history.append(ChatMessage(role=MessageRole.ASSISTANT, content="answer one"))

    next_msg, chat_kwargs, _ = await prepare_chat_params(
        "gemini-2.5-flash", list(history), "inline", None
    )

    assert _texts(history[0]) == ["question one"]

    request_texts = [
        part.text
        for content in chat_kwargs["history"]
        for part in (content.parts or [])
        if part.text
    ] + [part.text for part in (next_msg.parts or []) if part.text]
    assert request_texts.count("question two") == 1
