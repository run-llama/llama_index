from llama_index.voice_agents.elevenlabs.events import (
    PingEvent,
    AudioEvent,
    AgentResponseEvent,
    AgentResponseCorrectionEvent,
    UserTranscriptionEvent,
    InterruptionEvent,
    ConversationInitEvent,
    ClientToolCallEvent,
)
from llama_index.core.voice_agents import BaseVoiceAgentEvent


def test_events_init() -> None:
    events = [
        PingEvent(type_t="ping", ping_ms=100),
        AudioEvent(type_t="audio", base_64_encoded_audio="audio"),
        AgentResponseCorrectionEvent(
            type_t="agent_response_correction",
            corrected_agent_response="Corrected Response.",
        ),
        AgentResponseEvent(type_t="agent_response", agent_response="Response."),
        UserTranscriptionEvent(
            type_t="user_transcription", user_transcript="Transcript."
        ),
        InterruptionEvent(
            type_t="interruption", interrupted=True
        ),  # this tests the extra fields allowséd :)
        ConversationInitEvent(
            type_t="conversation_initiation_metadata",
            conversation_id="1",
            metadata={"latency": 12},
        ),
        ClientToolCallEvent(
            type_t="client_tool_call",
            tool_call_id="1",
            tool_name="greet",
            parameters={},
        ),
    ]
    for event in events:
        assert isinstance(event, BaseVoiceAgentEvent)
