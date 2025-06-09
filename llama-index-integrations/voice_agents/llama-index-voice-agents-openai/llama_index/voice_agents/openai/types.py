import base64
import binascii

from typing import Union, Optional, List
from typing_extensions import Self
from llama_index.core.bridge.pydantic import BaseModel, Field, model_validator


class ConversationBaseEvent(BaseModel):
    type_t: str = Field(serialization_alias="type")


class ConversationResponse(BaseModel):
    modalities: List[str] = Field(default=["text", "audio"])
    instructions: str = Field(default="You are a helpful assistant")


class ConversationStartEvent(ConversationBaseEvent):
    response: ConversationResponse = Field(
        default_factory=ConversationResponse,
    )


class ConversationInputEvent(ConversationBaseEvent):
    audio: Union[bytes, str]

    @model_validator(mode="after")
    def validate_audio_input(self) -> Self:
        try:
            base64.b64decode(self.audio, validate=True)
        except binascii.Error:
            if isinstance(self.audio, bytes):
                self.audio = base64.b64encode(self.audio).decode("utf-8")
        return self


class ConversationDeltaEvent(ConversationBaseEvent):
    delta: Union[str, bytes]
    item_id: str


class ConversationDoneEvent(ConversationBaseEvent):
    item_id: str
    text: Optional[str] = Field(
        default=None,
    )
    transcript: Optional[str] = Field(
        default=None,
    )
