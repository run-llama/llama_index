from typing import List

import pytest

from llama_index.llms.base import ChatMessage, MessageRole
from llama_index.llms.llama_utils import (B_INST, B_SYS, BOS,
                                          DEFAULT_SYSTEM_PROMPT, E_INST, E_SYS,
                                          EOS, messages_to_prompt)


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
    

def test_first_chat(chat_messages_first_chat):
    prompt = messages_to_prompt(chat_messages_first_chat)
    assert prompt == (
        f"{BOS} {B_INST} {B_SYS} some system message {E_SYS} "
        f"test question {E_INST}"
    )
    
    
def test_first_chat_default(chat_messages_first_chat_no_system):
    prompt = messages_to_prompt(chat_messages_first_chat_no_system)
    assert prompt == (
        f"{BOS} {B_INST} {B_SYS} {DEFAULT_SYSTEM_PROMPT} {E_SYS} "
        f"test question {E_INST}"
    )


def test_second_chat(chat_messages_second_chat):
    prompt = messages_to_prompt(chat_messages_second_chat)
    assert prompt == (
        f"{BOS} {B_INST} {B_SYS} some system message {E_SYS} "
        f"test question 1 {E_INST} some assistant reply {EOS} "
        f"{BOS} {B_INST} test question 2 {E_INST}"
    )


def test_second_chat_default(chat_messages_second_chat_no_system):
    prompt = messages_to_prompt(chat_messages_second_chat_no_system)
    assert prompt == (
        f"{BOS} {B_INST} {B_SYS} {DEFAULT_SYSTEM_PROMPT} {E_SYS} "
        f"test question 1 {E_INST} some assistant reply {EOS} "
        f"{BOS} {B_INST} test question 2 {E_INST}"
    )


def test_third_chat(chat_messages_third_chat):
    prompt = messages_to_prompt(chat_messages_third_chat)
    assert prompt == (
        f"{BOS} {B_INST} {B_SYS} some system message {E_SYS} "
        f"test question 1 {E_INST} some assistant reply 1 {EOS} "
        f"{BOS} {B_INST} test question 2 {E_INST} some assistant reply 2 {EOS} "
        f"{BOS} {B_INST} test question 3 {E_INST}"
    )


def test_third_chat_default(chat_messages_third_chat_no_system):
    prompt = messages_to_prompt(chat_messages_third_chat_no_system)
    assert prompt == (
        f"{BOS} {B_INST} {B_SYS} {DEFAULT_SYSTEM_PROMPT} {E_SYS} "
        f"test question 1 {E_INST} some assistant reply 1 {EOS} "
        f"{BOS} {B_INST} test question 2 {E_INST} some assistant reply 2 {EOS} "
        f"{BOS} {B_INST} test question 3 {E_INST}"
    )
