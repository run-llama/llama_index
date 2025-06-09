from llama_index.voice_agents.openai.types import (
    ConversationDeltaEvent,
    ConversationDoneEvent,
    ConversationResponse,
    ConversationStartEvent,
)


def test_serialization() -> None:
    start_event = ConversationStartEvent(
        type_t="response.create",
        response=ConversationResponse(instructions="Some instructions"),
    )
    assert start_event.model_dump(by_alias=True) == {
        "type": "response.create",
        "response": {
            "modalities": ["text", "audio"],
            "instructions": "Some instructions",
        },
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
