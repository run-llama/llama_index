from typing import Annotated

from enum import Enum
from llama_index.core.callbacks.base import CallbackManager
from llama_index.core.bridge.pydantic import WithJsonSchema


class PydanticAnnotations(Enum):
    CALLBACK_MANAGER = Annotated[
        CallbackManager,
        WithJsonSchema({"type": "object"}, mode="serialization"),
        WithJsonSchema({"type": "object"}, mode="validation"),
    ]


__all__ = ["PydanticAnnotations"]
