from typing import Any, Callable, List, Optional, Union

from chonkie.chunker.base import BaseChunker
from chonkie.pipeline import ComponentRegistry, ComponentType


from pydantic import Field

from llama_index.core.callbacks.base import CallbackManager
from llama_index.core.node_parser.interface import (
    MetadataAwareTextSplitter,
)
from llama_index.core.node_parser.node_utils import default_id_func

# a list of strings of all available chunkers in chonkie
# see https://github.com/chonkie-inc/chonkie/blob/cd8bd643bd7045686f0a8b73a64f1c9296c0dae2/src/chonkie/cli/cli_utils.py#L34-L36
CHUNKERS = sorted(
    c.alias
    for c in ComponentRegistry.list_components(component_type=ComponentType.CHUNKER)
    if c.alias not in ["table", "slumber"]
)


class Chunker(MetadataAwareTextSplitter):
    """
    Wrapper for Chonkie's chunkers.

    This class integrates Chonkie's chunking functionality with LlamaIndex's
    MetadataAwareTextSplitter interface.
    """

    # this is related to the metadata schema in the super, or pydantic will fail
    #  attributes need to be defined as pydantic fields
    chunker: Optional[BaseChunker] = Field(default=None, exclude=True)

    def __init__(
        self,
        chunker: Union[str, BaseChunker] = "recursive",
        callback_manager: Optional[CallbackManager] = None,
        include_metadata: bool = True,
        include_prev_next_rel: bool = True,
        id_func: Optional[Callable] = None,
        **kwargs: Any,
    ):
        if isinstance(chunker, str) and chunker not in CHUNKERS:
            raise ValueError(f"Invalid chunker '{chunker}'. Must be one of: {CHUNKERS}")

        id_func = id_func or default_id_func
        callback_manager = callback_manager or CallbackManager([])
        super().__init__(
            callback_manager=callback_manager,
            include_metadata=include_metadata,
            include_prev_next_rel=include_prev_next_rel,
            id_func=id_func,
        )

        if isinstance(chunker, str):
            # flexible approach to pull chunker classes based on their alias
            ChunkingClass = ComponentRegistry.get_chunker(chunker).component_class
            self.chunker = ChunkingClass(**kwargs)
        else:
            self.chunker = chunker

    @classmethod
    def from_defaults(
        cls,
        callback_manager: Optional[CallbackManager] = None,
        include_metadata: bool = True,
        include_prev_next_rel: bool = True,
    ) -> "Chunker":
        """Initialize with parameters."""
        callback_manager = callback_manager or CallbackManager([])
        return cls(
            callback_manager=callback_manager,
            include_metadata=include_metadata,
            include_prev_next_rel=include_prev_next_rel,
        )

    @classmethod
    def class_name(cls) -> str:
        return "Chunker"

    def split_text_metadata_aware(self, text: str, metadata_str: str) -> List[str]:
        """Split text with metadata awareness."""
        return self.split_text(text)

    def split_text(self, text: str) -> List[str]:
        """Split incoming text and return chunks using Chonkie chunker."""
        if text == "":
            return [text]

        if self.chunker is None:
            raise ValueError("Chunker not initialized")
        chunks = self.chunker.chunk(text)

        # extract attributes from chonkie Chunk dataclass
        # see https://github.com/chonkie-inc/chonkie/blob/cd8bd643bd7045686f0a8b73a64f1c9296c0dae2/src/chonkie/types/base.py#L32-L38
        if isinstance(chunks, list):
            return [
                chunk.text if hasattr(chunk, "text") else str(chunk) for chunk in chunks
            ]
        else:
            return [chunks.text if hasattr(chunks, "text") else str(chunks)]


# MonkeyPatch for https://github.com/run-llama/llama_index/pull/20622#discussion_r2764697454
Chunker.__init__.__doc__ = f"""
        Initialize with a Chonkie chunker instance or create one if not provided.

        Args:
            chunker Union[str, BaseChunker]: The chunker to use. Must be one of {CHUNKERS} or a chonkie chunker instance.
            callback_manager (Optional[CallbackManager]): Callback manager for handling callbacks.
            include_metadata (bool): Whether to include metadata in the nodes.
            include_prev_next_rel (bool): Whether to include previous/next relationships.
            id_func (Optional[Callable]): Function to generate node IDs.
            **kwargs: Additional keyword arguments for Chonkie's RecursiveChunker.

        """
