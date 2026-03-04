from llama_index.core.voice_agents import BaseVoiceAgentEvent
from typing import Dict, Any


class AudioSentEvent(BaseVoiceAgentEvent):
    data: bytes


class AudioReceivedEvent(BaseVoiceAgentEvent):
    data: bytes


class TextSentEvent(BaseVoiceAgentEvent):
    text: str


class TextReceivedEvent(BaseVoiceAgentEvent):
    text: str


class ToolCallEvent(BaseVoiceAgentEvent):
    tool_name: str
    tool_args: Dict[str, Any]


class ToolCallResultEvent(BaseVoiceAgentEvent):
    tool_name: str
    tool_result: Any
