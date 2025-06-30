import pytest

from llama_index.voice_agents.openai.types import (
    ConversationDeltaEvent,
    ConversationDoneEvent,
    ConversationSession,
    ConversationSessionUpdate,
)


@pytest.fixture()
def session_json() -> dict:
    return {
        "modalities": ["text", "audio"],
        "instructions": "You are a helpful assistant.",
        "voice": "sage",
        "input_audio_format": "pcm16",
        "output_audio_format": "pcm16",
        "input_audio_transcription": {"model": "whisper-1"},
        "turn_detection": {
            "type": "server_vad",
            "threshold": 0.5,
            "prefix_padding_ms": 300,
            "silence_duration_ms": 500,
            "create_response": True,
        },
        "tools": [],
        "tool_choice": "auto",
        "temperature": 0.8,
        "max_response_output_tokens": "inf",
        "speed": 1.1,
        "tracing": "auto",
    }


def test_serialization(session_json: dict) -> None:
    start_event = ConversationSessionUpdate(
        type_t="session.update",
        session=ConversationSession(),
    )
    assert start_event.model_dump(by_alias=True) == {
        "type": "session.update",
        "session": session_json,
    }


def test_deserialization() -> None:
    message = {"type_t": "response.text.delta", "delta": "Hello", "item_id": "msg_001"}
    assert ConversationDeltaEvent.model_validate(message) == ConversationDeltaEvent(
        type_t="response.text.delta", delta="Hello", item_id="msg_001"
    )
    message1 = {
        "type_t": "response.text.done",
        "text": "Hello world, this is a test!",
        "item_id": "msg_001",
    }
    assert ConversationDoneEvent.model_validate(message1) == ConversationDoneEvent(
        type_t="response.text.done",
        text="Hello world, this is a test!",
        item_id="msg_001",
    )
    message2 = {
        "type_t": "response.audio_transcript.done",
        "transcript": "Hello world, this is a test!",
        "item_id": "msg_001",
    }
    assert ConversationDoneEvent.model_validate(message2) == ConversationDoneEvent(
        type_t="response.audio_transcript.done",
        transcript="Hello world, this is a test!",
        item_id="msg_001",
    )
