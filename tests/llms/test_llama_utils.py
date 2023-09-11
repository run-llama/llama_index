from typing import Sequence

import pytest

from llama_index.llms.base import ChatMessage, MessageRole
from llama_index.llms.llama_utils import (
    B_INST,
    B_SYS,
    BOS,
    DEFAULT_SYSTEM_PROMPT,
    E_INST,
    E_SYS,
    EOS,
    completion_to_prompt,
    messages_to_prompt,
)


@pytest.fixture
def chat_messages_first_chat() -> Sequence[ChatMessage]:
    # example first chat with system message
    return [
        ChatMessage(role=MessageRole.SYSTEM, content="some system message"),
        ChatMessage(role=MessageRole.USER, content="test question"),
    ]


@pytest.fixture
def chat_messages_first_chat_no_system(
    chat_messages_first_chat: Sequence[ChatMessage],
) -> Sequence[ChatMessage]:
    # example first chat without system message
    return chat_messages_first_chat[1:]


@pytest.fixture
def chat_messages_second_chat() -> Sequence[ChatMessage]:
    # example second chat with system message
    return [
        ChatMessage(role=MessageRole.SYSTEM, content="some system message"),
        ChatMessage(role=MessageRole.USER, content="test question 1"),
        ChatMessage(role=MessageRole.ASSISTANT, content="some assistant reply"),
        ChatMessage(role=MessageRole.USER, content="test question 2"),
    ]


@pytest.fixture
def chat_messages_second_chat_no_system(
    chat_messages_second_chat: Sequence[ChatMessage],
) -> Sequence[ChatMessage]:
    # example second chat without system message
    return chat_messages_second_chat[1:]


@pytest.fixture
def chat_messages_third_chat() -> Sequence[ChatMessage]:
    # example third chat with system message
    return [
        ChatMessage(role=MessageRole.SYSTEM, content="some system message"),
        ChatMessage(role=MessageRole.USER, content="test question 1"),
        ChatMessage(role=MessageRole.ASSISTANT, content="some assistant reply 1"),
        ChatMessage(role=MessageRole.USER, content="test question 2"),
        ChatMessage(role=MessageRole.ASSISTANT, content="some assistant reply 2"),
        ChatMessage(role=MessageRole.USER, content="test question 3"),
    ]


@pytest.fixture
def chat_messages_third_chat_no_system(
    chat_messages_third_chat: Sequence[ChatMessage],
) -> Sequence[ChatMessage]:
    # example third chat without system message
    return chat_messages_third_chat[1:]


@pytest.fixture
def chat_messages_assistant_first() -> Sequence[ChatMessage]:
    # assistant message first in chat (after system)
    # should raise error as we expect the first message after any system
    # message to be a user message
    return [
        ChatMessage(role=MessageRole.SYSTEM, content="some system message"),
        ChatMessage(role=MessageRole.ASSISTANT, content="some assistant reply"),
        ChatMessage(role=MessageRole.USER, content="test question"),
    ]


@pytest.fixture
def chat_messages_user_twice() -> Sequence[ChatMessage]:
    # user message twice in a row (after system)
    # should raise error as we expect an assistant message
    # to follow a user message
    return [
        ChatMessage(role=MessageRole.SYSTEM, content="some system message"),
        ChatMessage(role=MessageRole.USER, content="test question 1"),
        ChatMessage(role=MessageRole.USER, content="test question 2"),
    ]


def test_first_chat(chat_messages_first_chat: Sequence[ChatMessage]) -> None:
    # test first chat prompt creation with system prompt
    prompt = messages_to_prompt(chat_messages_first_chat)
    assert prompt == (
        f"{BOS} {B_INST} {B_SYS} some system message {E_SYS} test question {E_INST}"
    )


def test_first_chat_default(
    chat_messages_first_chat_no_system: Sequence[ChatMessage],
) -> None:
    # test first chat prompt creation without system prompt and use default
    prompt = messages_to_prompt(chat_messages_first_chat_no_system)
    assert prompt == (
        f"{BOS} {B_INST} {B_SYS} {DEFAULT_SYSTEM_PROMPT.strip()} {E_SYS} "
        f"test question {E_INST}"
    )


def test_second_chat(chat_messages_second_chat: Sequence[ChatMessage]) -> None:
    # test second chat prompt creation with system prompt
    prompt = messages_to_prompt(chat_messages_second_chat)
    assert prompt == (
        f"{BOS} {B_INST} {B_SYS} some system message {E_SYS} "
        f"test question 1 {E_INST} some assistant reply {EOS}"
        f"{BOS} {B_INST} test question 2 {E_INST}"
    )


def test_second_chat_default(
    chat_messages_second_chat_no_system: Sequence[ChatMessage],
) -> None:
    # test second chat prompt creation without system prompt and use default
    prompt = messages_to_prompt(chat_messages_second_chat_no_system)
    assert prompt == (
        f"{BOS} {B_INST} {B_SYS} {DEFAULT_SYSTEM_PROMPT.strip()} {E_SYS} "
        f"test question 1 {E_INST} some assistant reply {EOS}"
        f"{BOS} {B_INST} test question 2 {E_INST}"
    )


def test_third_chat(chat_messages_third_chat: Sequence[ChatMessage]) -> None:
    # test third chat prompt creation with system prompt
    prompt = messages_to_prompt(chat_messages_third_chat)
    assert prompt == (
        f"{BOS} {B_INST} {B_SYS} some system message {E_SYS} "
        f"test question 1 {E_INST} some assistant reply 1 {EOS}"
        f"{BOS} {B_INST} test question 2 {E_INST} some assistant reply 2 {EOS}"
        f"{BOS} {B_INST} test question 3 {E_INST}"
    )


def test_third_chat_default(
    chat_messages_third_chat_no_system: Sequence[ChatMessage],
) -> None:
    # test third chat prompt creation without system prompt and use default
    prompt = messages_to_prompt(chat_messages_third_chat_no_system)
    assert prompt == (
        f"{BOS} {B_INST} {B_SYS} {DEFAULT_SYSTEM_PROMPT.strip()} {E_SYS} "
        f"test question 1 {E_INST} some assistant reply 1 {EOS}"
        f"{BOS} {B_INST} test question 2 {E_INST} some assistant reply 2 {EOS}"
        f"{BOS} {B_INST} test question 3 {E_INST}"
    )


def test_error_assistant_first(
    chat_messages_assistant_first: Sequence[ChatMessage],
) -> None:
    # should have error if assistant message occurs first
    with pytest.raises(AssertionError):
        messages_to_prompt(chat_messages_assistant_first)


def test_error_user_twice(chat_messages_user_twice: Sequence[ChatMessage]) -> None:
    # should have error if second message is user
    # (or have user twice in a row)
    with pytest.raises(AssertionError):
        messages_to_prompt(chat_messages_user_twice)


def test_completion_to_prompt() -> None:
    # test prompt creation from completion with system prompt
    completion = "test completion"
    system_prompt = "test system prompt"
    prompt = completion_to_prompt(completion, system_prompt=system_prompt)
    assert prompt == (
        f"{BOS} {B_INST} {B_SYS} {system_prompt} {E_SYS} {completion} {E_INST}"
    )


def test_completion_to_prompt_default() -> None:
    # test prompt creation from completion without system prompt and use default
    completion = "test completion"
    prompt = completion_to_prompt(completion)
    assert prompt == (
        f"{BOS} {B_INST} {B_SYS} {DEFAULT_SYSTEM_PROMPT.strip()} {E_SYS} "
        f"{completion} {E_INST}"
    )
