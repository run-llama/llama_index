import base64
import binascii
import json

from typing import Union, List, Dict, Literal, Optional, Any
from typing_extensions import Self
from llama_index.core.voice_agents import BaseVoiceAgentEvent
from llama_index.core.bridge.pydantic import BaseModel, Field, model_validator


class ConversationVAD(BaseModel):
    type_t: str = Field(serialization_alias="type", default="server_vad")
    threshold: float = Field(default=0.5)
    prefix_padding_ms: int = Field(default=300)
    silence_duration_ms: int = Field(default=500)
    create_response: bool = Field(default=True)


class ParamPropertyDefinition(BaseModel):
    type: str


class ToolParameters(BaseModel):
    type: Literal["object"] = Field(default="object")
    properties: Dict[str, ParamPropertyDefinition]
    required: List[str]


class FunctionResultItem(BaseVoiceAgentEvent):
    call_id: str
    output: str


class ConversationTool(BaseModel):
    type: Literal["function"] = Field(default="function")
    name: str
    description: str
    parameters: ToolParameters


class SendFunctionItemEvent(BaseVoiceAgentEvent):
    item: FunctionResultItem


class ConversationSession(BaseModel):
    modalities: List[str] = Field(default=["text", "audio"])
    instructions: str = Field(default="You are a helpful assistant.")
    voice: str = Field(default="sage")
    input_audio_format: str = Field(default="pcm16")
    output_audio_format: str = Field(default="pcm16")
    input_audio_transcription: Dict[Literal["model"], str] = Field(
        max_length=1, default={"model": "whisper-1"}
    )
    turn_detection: ConversationVAD = Field(default_factory=ConversationVAD)
    tools: List[ConversationTool] = Field(
        default_factory=list,
    )
    tool_choice: Literal["auto", "none", "required"] = Field(default="auto")
    temperature: float = Field(default=0.8, ge=0.6)
    max_response_output_tokens: Union[Literal["inf"], int] = Field(
        default="inf",
        ge=1,
        le=4096,
    )
    speed: float = Field(default=1.1)
    tracing: Union[Literal["auto"], Dict] = Field(default="auto")


class ConversationSessionUpdate(BaseVoiceAgentEvent):
    session: ConversationSession


class ConversationInputEvent(BaseVoiceAgentEvent):
    audio: Union[bytes, str]

    @model_validator(mode="after")
    def validate_audio_input(self) -> Self:
        try:
            base64.b64decode(self.audio, validate=True)
        except binascii.Error:
            if isinstance(self.audio, bytes):
                self.audio = base64.b64encode(self.audio).decode("utf-8")
        return self


class FunctionCallDoneEvent(BaseVoiceAgentEvent):
    call_id: str
    name: Optional[str] = Field(default=None)
    arguments: Union[str, Dict[str, Any]]
    item_id: str

    @model_validator(mode="after")
    def validate_arguments(self) -> Self:
        try:
            self.arguments = json.loads(self.arguments)
        except json.JSONDecodeError:
            raise ValueError("arguments are non-serializable")
        return self


class ConversationDeltaEvent(BaseVoiceAgentEvent):
    delta: Union[str, bytes]
    item_id: str


class ConversationDoneEvent(BaseVoiceAgentEvent):
    item_id: str
    text: Optional[str] = Field(
        default=None,
    )
    transcript: Optional[str] = Field(
        default=None,
    )
