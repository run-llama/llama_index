import pytest

from llama_index.core.llms import ChatMessage, TextBlock
from llama_index.core.agent.utils import messages_to_xml_format
from typing import List


@pytest.fixture
def chat_messages() -> List[ChatMessage]:
    return [
        ChatMessage(role="user", blocks=[TextBlock(text="hello")]),
        ChatMessage(role="assistant", blocks=[TextBlock(text="hello back")]),
        ChatMessage(role="user", blocks=[TextBlock(text="how are you?")]),
        ChatMessage(role="assistant", blocks=[TextBlock(text="I am good, thank you.")]),
    ]


@pytest.fixture
def xml_string() -> str:
    return "<current_conversation>\n\t<user>\n\t\t<message>hello</message>\n\t</user>\n\t<assistant>\n\t\t<message>hello back</message>\n\t</assistant>\n\t<user>\n\t\t<message>how are you?</message>\n\t</user>\n\t<assistant>\n\t\t<message>I am good, thank you.</message>\n\t</assistant>\n</current_conversation>\n\nGiven the conversation, format the output according to the provided schema."


def test_messages_to_xml(chat_messages: List[ChatMessage], xml_string: str):
    msg = messages_to_xml_format(chat_messages)
    assert isinstance(msg, ChatMessage)
    s = ""
    for block in msg.blocks:
        s += block.text
    assert s == xml_string
