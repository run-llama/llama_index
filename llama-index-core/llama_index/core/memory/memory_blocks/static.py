from typing import Any, List, Optional, Union

from llama_index.core.base.llms.types import ChatMessage, ContentBlock, TextBlock
from llama_index.core.bridge.pydantic import Field, field_validator
from llama_index.core.memory.memory import BaseMemoryBlock


class StaticMemoryBlock(BaseMemoryBlock[List[ContentBlock]]):
    """
    A memory block that returns static text.

    This block is useful for including constant information or instructions
    in the context without relying on external processing.
    """

    name: str = Field(
        default="StaticContent", description="The name of the memory block."
    )
    static_content: Union[List[ContentBlock]] = Field(
        description="Static text or content to be returned by this memory block."
    )

    @field_validator("static_content", mode="before")
    @classmethod
    def validate_static_content(
        cls, v: Union[str, List[ContentBlock]]
    ) -> List[ContentBlock]:
        if isinstance(v, str):
            v = [TextBlock(text=v)]
        return v

    async def _aget(
        self, messages: Optional[List[ChatMessage]] = None, **block_kwargs: Any
    ) -> List[ContentBlock]:
        """Return the static text, potentially filtered by conditions."""
        return self.static_content

    async def _aput(self, messages: List[ChatMessage]) -> None:
        """No-op for static blocks as they don't change."""
