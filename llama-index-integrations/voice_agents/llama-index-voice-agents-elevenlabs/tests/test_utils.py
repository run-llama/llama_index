import pytest

from typing import Dict, List
from llama_index.core.llms import ChatMessage, MessageRole, TextBlock, AudioBlock
from llama_index.voice_agents.elevenlabs.utils import (
    callback_agent_message,
    callback_agent_message_correction,
    callback_latency_measurement,
    callback_user_message,
)

data = b"fake_audio_data"


@pytest.fixture()
def messages() -> Dict[int, List[ChatMessage]]:
    return {
        1: [
            ChatMessage(role=MessageRole.ASSISTANT, blocks=[AudioBlock(audio=data)]),
            ChatMessage(role=MessageRole.USER, blocks=[TextBlock(text="Hello")]),
        ]
    }


@pytest.fixture()
def latencies() -> List[int]:
    return [1, 3]


def test_agent_message(messages: Dict[int, List[ChatMessage]]):
    local_messages = messages.copy()
    callback_agent_message(messages=local_messages, message_id=1, text="Hello")
    assert {
        1: [
            ChatMessage(
                role=MessageRole.ASSISTANT,
                blocks=[AudioBlock(audio=data), TextBlock(text="Hello")],
            ),
            ChatMessage(role=MessageRole.USER, blocks=[TextBlock(text="Hello")]),
        ]
    } == local_messages
    callback_agent_message(messages=local_messages, message_id=2, text="Hello")
    assert {
        1: [
            ChatMessage(
                role=MessageRole.ASSISTANT,
                blocks=[AudioBlock(audio=data), TextBlock(text="Hello")],
            ),
            ChatMessage(role=MessageRole.USER, blocks=[TextBlock(text="Hello")]),
        ],
        2: [ChatMessage(role=MessageRole.ASSISTANT, blocks=[TextBlock(text="Hello")])],
    } == local_messages
    callback_agent_message(messages=local_messages, message_id=2, audio=data)
    assert {
        1: [
            ChatMessage(
                role=MessageRole.ASSISTANT,
                blocks=[AudioBlock(audio=data), TextBlock(text="Hello")],
            ),
            ChatMessage(role=MessageRole.USER, blocks=[TextBlock(text="Hello")]),
        ],
        2: [
            ChatMessage(
                role=MessageRole.ASSISTANT,
                blocks=[TextBlock(text="Hello"), AudioBlock(audio=data)],
            )
        ],
    } == local_messages


def test_user_message(messages: Dict[int, List[ChatMessage]]):
    local_messages = messages.copy()
    callback_user_message(messages=local_messages, message_id=1, audio=data)
    assert {
        1: [
            ChatMessage(role=MessageRole.ASSISTANT, blocks=[AudioBlock(audio=data)]),
            ChatMessage(
                role=MessageRole.USER,
                blocks=[TextBlock(text="Hello"), AudioBlock(audio=data)],
            ),
        ]
    } == local_messages
    callback_user_message(messages=local_messages, message_id=2, text="Hello")
    assert {
        1: [
            ChatMessage(role=MessageRole.ASSISTANT, blocks=[AudioBlock(audio=data)]),
            ChatMessage(
                role=MessageRole.USER,
                blocks=[TextBlock(text="Hello"), AudioBlock(audio=data)],
            ),
        ],
        2: [ChatMessage(role=MessageRole.USER, blocks=[TextBlock(text="Hello")])],
    } == local_messages
    callback_user_message(messages=local_messages, message_id=2, audio=data)
    assert {
        1: [
            ChatMessage(role=MessageRole.ASSISTANT, blocks=[AudioBlock(audio=data)]),
            ChatMessage(
                role=MessageRole.USER,
                blocks=[TextBlock(text="Hello"), AudioBlock(audio=data)],
            ),
        ],
        2: [
            ChatMessage(
                role=MessageRole.USER,
                blocks=[TextBlock(text="Hello"), AudioBlock(audio=data)],
            )
        ],
    } == local_messages


def test_agent_message_correction(messages: Dict[int, List[ChatMessage]]):
    local_messages = messages.copy()
    local_messages[1][0].blocks.append(TextBlock(text="Hell"))
    callback_agent_message_correction(
        messages=local_messages, message_id=1, text="Hello"
    )
    assert local_messages[1][0].blocks[1].text == "Hello"


def test_latencies(latencies: List[int]):
    local_lats = latencies.copy()
    callback_latency_measurement(local_lats, 3)
    callback_latency_measurement(local_lats, 9)
    assert local_lats == [*latencies, 3, 9]
