from llama_index.llms.litellm.utils import (
    openai_modelname_to_contextsize,
    to_openai_message_dicts,
)
from llama_index.core.base.llms.types import ChatMessage, MessageRole


def test_model_context_size():
    assert openai_modelname_to_contextsize("gpt-4") == 4096
    assert openai_modelname_to_contextsize("gpt-3.5-turbo") == 4096
    assert openai_modelname_to_contextsize("unknown-model") == 2048


def test_message_conversion():
    # Test converting to OpenAI message format
    messages = [
        ChatMessage(role=MessageRole.USER, content="Hello"),
        ChatMessage(role=MessageRole.ASSISTANT, content="Hi there!"),
        ChatMessage(role=MessageRole.SYSTEM, content="Be helpful"),
    ]

    openai_messages = to_openai_message_dicts(messages)
    assert len(openai_messages) == 3
    assert openai_messages[0]["role"] == "user"
    assert openai_messages[0]["content"] == "Hello"
    assert openai_messages[1]["role"] == "assistant"
    assert openai_messages[2]["role"] == "system"
