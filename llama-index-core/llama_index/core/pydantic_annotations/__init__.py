from typing import Annotated, Optional

from enum import Enum
from llama_index.core.callbacks.base import CallbackManager
from llama_index.core.bridge.pydantic import WithJsonSchema


class PydanticAnnotations(Enum):
    CALLBACK_MANAGER = Annotated[
        Optional[CallbackManager],
        WithJsonSchema({"type": "object"}, mode="serialization"),
        WithJsonSchema({"type": "object"}, mode="validation"),
    ]


__all__ = ["PydanticAnnotations"]
