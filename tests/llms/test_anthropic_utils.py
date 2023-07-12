from llama_index.llms.base import ChatMessage, MessageRole
from llama_index.llms.anthropic_utils import messages_to_anthropic_prompt


def test_messages_to_anthropic_prompt() -> None:
    messages = [
        ChatMessage(role=MessageRole.USER, content="Hello"),
    ]

    expected_prompt = "\n\nHuman: Hello\n\nAssistant: "
    actual_prompt = messages_to_anthropic_prompt(messages)
    assert actual_prompt == expected_prompt
