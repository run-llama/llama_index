from typing import List

import pytest

from llama_index.llms.base import ChatMessage, MessageRole
from llama_index.llms.llama_utils import (B_INST, B_SYS, BOS,
                                          DEFAULT_SYSTEM_PROMPT, E_INST, E_SYS,
                                          EOS, completion_to_prompt,
                                          messages_to_prompt)


@pytest.fixture
def chat_messages_first_chat() -> List[ChatMessage]:
    return [
        ChatMessage(role=MessageRole.SYSTEM, content="some system message"),
        ChatMessage(role=MessageRole.USER, content="test question"),
    ]
    
@pytest.fixture
def chat_messages_first_chat_no_system(chat_messages_first_chat) -> List[ChatMessage]:
    return chat_messages_first_chat[1:]
    
@pytest.fixture
def chat_messages_second_chat() -> List[ChatMessage]:
    return [
        ChatMessage(role=MessageRole.SYSTEM, content="some system message"),
        ChatMessage(role=MessageRole.USER, content="test question 1"),
        ChatMessage(role=MessageRole.ASSISTANT, content="some assistant reply"),
        ChatMessage(role=MessageRole.USER, content="test question 2"),
    ]
    
@pytest.fixture
def chat_messages_second_chat_no_system(chat_messages_second_chat) -> List[ChatMessage]:
    return chat_messages_second_chat[1:]

@pytest.fixture
def chat_messages_third_chat() -> List[ChatMessage]:
    return [
        ChatMessage(role=MessageRole.SYSTEM, content="some system message"),
        ChatMessage(role=MessageRole.USER, content="test question 1"),
        ChatMessage(role=MessageRole.ASSISTANT, content="some assistant reply 1"),
        ChatMessage(role=MessageRole.USER, content="test question 2"),
        ChatMessage(role=MessageRole.ASSISTANT, content="some assistant reply 2"),
        ChatMessage(role=MessageRole.USER, content="test question 3"),
    ]

@pytest.fixture
def chat_messages_third_chat_no_system(chat_messages_third_chat) -> List[ChatMessage]:
    return chat_messages_third_chat[1:]

@pytest.fixture
def chat_messages_assistant_first() -> List[ChatMessage]:
    # assistant message first in chat (after system)
    return [
        ChatMessage(role=MessageRole.SYSTEM, content="some system message"),
        ChatMessage(role=MessageRole.ASSISTANT, content="some assistant reply"),
        ChatMessage(role=MessageRole.USER, content="test question"),
    ]
    
@pytest.fixture
def chat_messages_user_twice() -> List[ChatMessage]:
    # user message twice in a row (after system)
    return [
        ChatMessage(role=MessageRole.SYSTEM, content="some system message"),
        ChatMessage(role=MessageRole.USER, content="test question 1"),
        ChatMessage(role=MessageRole.USER, content="test question 2"),
    ]
    

def test_first_chat(chat_messages_first_chat):
    prompt = messages_to_prompt(chat_messages_first_chat)
    assert prompt == (
        f"{BOS} {B_INST} {B_SYS} some system message {E_SYS} "
        f"test question {E_INST}"
    )
    
    
def test_first_chat_default(chat_messages_first_chat_no_system):
    prompt = messages_to_prompt(chat_messages_first_chat_no_system)
    assert prompt == (
        f"{BOS} {B_INST} {B_SYS} {DEFAULT_SYSTEM_PROMPT.strip()} {E_SYS} "
        f"test question {E_INST}"
    )


def test_second_chat(chat_messages_second_chat):
    prompt = messages_to_prompt(chat_messages_second_chat)
    assert prompt == (
        f"{BOS} {B_INST} {B_SYS} some system message {E_SYS} "
        f"test question 1 {E_INST} some assistant reply {EOS}"
        f"{BOS} {B_INST} test question 2 {E_INST}"
    )


def test_second_chat_default(chat_messages_second_chat_no_system):
    prompt = messages_to_prompt(chat_messages_second_chat_no_system)
    assert prompt == (
        f"{BOS} {B_INST} {B_SYS} {DEFAULT_SYSTEM_PROMPT.strip()} {E_SYS} "
        f"test question 1 {E_INST} some assistant reply {EOS}"
        f"{BOS} {B_INST} test question 2 {E_INST}"
    )


def test_third_chat(chat_messages_third_chat):
    prompt = messages_to_prompt(chat_messages_third_chat)
    assert prompt == (
        f"{BOS} {B_INST} {B_SYS} some system message {E_SYS} "
        f"test question 1 {E_INST} some assistant reply 1 {EOS}"
        f"{BOS} {B_INST} test question 2 {E_INST} some assistant reply 2 {EOS}"
        f"{BOS} {B_INST} test question 3 {E_INST}"
    )


def test_third_chat_default(chat_messages_third_chat_no_system):
    prompt = messages_to_prompt(chat_messages_third_chat_no_system)
    assert prompt == (
        f"{BOS} {B_INST} {B_SYS} {DEFAULT_SYSTEM_PROMPT.strip()} {E_SYS} "
        f"test question 1 {E_INST} some assistant reply 1 {EOS}"
        f"{BOS} {B_INST} test question 2 {E_INST} some assistant reply 2 {EOS}"
        f"{BOS} {B_INST} test question 3 {E_INST}"
    )


def test_error_assistant_first(chat_messages_assistant_first):
    # should have error if assistant message occurs first
    with pytest.raises(AssertionError):
        messages_to_prompt(chat_messages_assistant_first)


def test_error_user_twice(chat_messages_user_twice):
    # should have error if second message is user
    # (or have user twice in a row)
    with pytest.raises(AssertionError):
        messages_to_prompt(chat_messages_user_twice)
        

def test_completion_to_prompt():
    completion = "test completion"
    system_prompt = "test system prompt"
    prompt = completion_to_prompt(completion, system_prompt=system_prompt)
    assert prompt == (
        f"{BOS} {B_INST} {B_SYS} {system_prompt} {E_SYS} "
        f"{completion} {E_INST}"
    )


def test_completion_to_prompt_default():
    completion = "test completion"
    prompt = completion_to_prompt(completion)
    assert prompt == (
        f"{BOS} {B_INST} {B_SYS} {DEFAULT_SYSTEM_PROMPT.strip()} {E_SYS} "
        f"{completion} {E_INST}"
    )
