"""Extra types that might be helpful for future releases."""

from pydantic import BaseModel, Field
from typing import Union, List, Dict, Literal


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


class ConversationTool(BaseModel):
    type: Literal["function"] = Field(default="function")
    name: str
    description: str
    parameters: ToolParameters


class ConversationSession(BaseModel):
    modalities: List[str] = Field(default=["text", "audio"])
    instructions: str = Field(default="You are a helpful assistant")
    voice: str = Field(default="sage")
    input_audio_format: str = Field(default="pcm16")
    output_audio_format: str = Field(default="pcm16")
    input_audio_transcription: Dict[str, str] = Field(
        max_length=1, default={"model": "whisper-1"}
    )
    turn_detection: ConversationVAD = Field(default_factory=ConversationVAD)
    tools: List[ConversationTool] = Field(
        default_factory=list,
    )
    tool_choice: Literal["auto", "none", "required"] = Field(default="auto")
    temperature: float = Field(default=0.5)
    max_response_output_tokens: Union[Literal["inf"], int] = Field(
        default="inf",
        ge=1,
        le=4096,
    )
    speed: float = Field(default=1.1)
    tracing: Union[Literal["auto"], Dict] = Field(default="auto")
