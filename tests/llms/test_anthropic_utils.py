import pytest

from llama_index.llms.anthropic_utils import (
    anthropic_modelname_to_contextsize,
    messages_to_anthropic_prompt,
)
from llama_index.llms.base import ChatMessage, MessageRole


def test_messages_to_anthropic_prompt() -> None:
    messages = [
        ChatMessage(role=MessageRole.USER, content="Hello"),
    ]

    expected_prompt = "\n\nHuman: Hello\n\nAssistant: "
    actual_prompt = messages_to_anthropic_prompt(messages)
    assert actual_prompt == expected_prompt

    messages = [
        ChatMessage(role=MessageRole.USER, content="Hello"),
        ChatMessage(role=MessageRole.ASSISTANT, content="Continue this sentence"),
    ]

    expected_prompt = "\n\nHuman: Hello\n\nAssistant: Continue this sentence"
    actual_prompt = messages_to_anthropic_prompt(messages)
    assert actual_prompt == expected_prompt


def test_anthropic_modelname_to_contextsize() -> None:
    with pytest.raises(ValueError):
        anthropic_modelname_to_contextsize("bad name")
