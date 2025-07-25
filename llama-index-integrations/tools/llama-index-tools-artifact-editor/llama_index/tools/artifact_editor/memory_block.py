from typing import List, Optional, Any

from llama_index.core.bridge.pydantic import Field
from llama_index.core.llms import ChatMessage
from llama_index.core.memory import BaseMemoryBlock
from llama_index.tools.artifact_editor.base import ArtifactEditorToolSpec


class ArtifactMemoryBlock(BaseMemoryBlock[str]):
    """Custom memory block to maintain the artifact in-memory."""

    name: str = Field(
        default="current_artifact", description="The name of the artifact block"
    )
    artifact_spec: Optional[ArtifactEditorToolSpec] = Field(
        default=None, description="The artifact spec for the artifact block"
    )

    async def _aget(
        self, messages: Optional[List[ChatMessage]] = None, **kwargs: Any
    ) -> str:
        if self.artifact_spec.get_current_artifact() is None:
            return "No artifact created yet"
        return str(self.artifact_spec.get_current_artifact())

    async def _aput(self, messages: List[ChatMessage]) -> None:
        pass
