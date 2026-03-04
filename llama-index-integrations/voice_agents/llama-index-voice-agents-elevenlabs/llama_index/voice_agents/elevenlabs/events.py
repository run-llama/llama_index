from llama_index.core.voice_agents import BaseVoiceAgentEvent
from llama_index.core.bridge.pydantic import ConfigDict

from typing import Union, Any


class ConversationInitEvent(BaseVoiceAgentEvent):
    model_config = ConfigDict(extra="allow")
    conversation_id: str


class AudioEvent(BaseVoiceAgentEvent):
    model_config = ConfigDict(extra="allow")
    base_64_encoded_audio: str


class AgentResponseEvent(BaseVoiceAgentEvent):
    model_config = ConfigDict(extra="allow")
    agent_response: str


class AgentResponseCorrectionEvent(BaseVoiceAgentEvent):
    model_config = ConfigDict(extra="allow")
    corrected_agent_response: str


class UserTranscriptionEvent(BaseVoiceAgentEvent):
    model_config = ConfigDict(extra="allow")
    user_transcript: str


class InterruptionEvent(BaseVoiceAgentEvent):
    model_config = ConfigDict(extra="allow")


class PingEvent(BaseVoiceAgentEvent):
    model_config = ConfigDict(extra="allow")
    ping_ms: Union[float, int]


class ClientToolCallEvent(BaseVoiceAgentEvent):
    model_config = ConfigDict(extra="allow")
    tool_call_id: str
    tool_name: str
    parameters: Any
