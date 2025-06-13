from llama_index.core.bridge.pydantic import BaseModel, Field


class BaseVoiceAgentEvent(BaseModel):
    """
    Base class to represent events in Voice Agents conversations.

    Attributes:
        type_t (str): Event type (serialized as 'type')

    """

    type_t: str = Field(serialization_alias="type")
