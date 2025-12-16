"""Token splitter."""

import logging
from collections.abc import Callable


from llama_index.core.base.llms.types import (
    ChatMessage,
    TextBlock,
)
from llama_index.core.bridge.pydantic import Field, PrivateAttr
from llama_index.core.callbacks.base import CallbackManager
from llama_index.core.callbacks.schema import CBEventType, EventPayload
from llama_index.core.constants import DEFAULT_CHUNK_OVERLAP, DEFAULT_CHUNK_SIZE
from llama_index.core.node_parser.interface import MetadataAwareMessageSplitter
from llama_index.core.node_parser.node_utils import default_id_func
from llama_index.core.node_parser.text.utils import split_by_char, split_by_sep
from llama_index.core.schema import Document
from llama_index.core.utils import get_tokenizer


_logger = logging.getLogger(__name__)

# NOTE: this is the number of tokens we reserve for metadata formatting
DEFAULT_METADATA_FORMAT_LEN = 2


class TokenMessageSplitter(MetadataAwareMessageSplitter):
    """Implementation of splitting text that looks at word tokens."""

    chunk_size: int = Field(
        default=DEFAULT_CHUNK_SIZE,
        description="The token chunk size for each chunk.",
        gt=0,
    )
    chunk_overlap: int = Field(
        default=DEFAULT_CHUNK_OVERLAP,
        description="The token overlap of each chunk when splitting.",
        ge=0,
    )
    separator: str = Field(
        default=" ", description="Default separator for splitting into words"
    )
    backup_separators: list = Field(
        default_factory=list, description="Additional separators for splitting."
    )

    keep_whitespaces: bool = Field(
        default=False,
        description="Whether to keep leading/trailing whitespaces in the chunk.",
    )

    _tokenizer: Callable = PrivateAttr()
    _split_fns: list[Callable] = PrivateAttr()

    def __init__(
        self,
        chunk_size: int = DEFAULT_CHUNK_SIZE,
        chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
        tokenizer: Callable | None = None,
        callback_manager: CallbackManager | None = None,
        separator: str = " ",
        backup_separators: list[str] | None = ["\n"],
        keep_whitespaces: bool = False,
        include_metadata: bool = True,
        include_prev_next_rel: bool = True,
        id_func: Callable[[int, Document], str] | None = None,
    ):
        """Initialize with parameters."""
        if chunk_overlap > chunk_size:
            raise ValueError(
                f"Got a larger chunk overlap ({chunk_overlap}) than chunk size ({chunk_size}), should be smaller."
            )
        callback_manager = callback_manager or CallbackManager([])
        id_func = id_func or default_id_func
        super().__init__(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separator=separator,
            backup_separators=backup_separators,
            keep_whitespaces=keep_whitespaces,
            callback_manager=callback_manager,
            include_metadata=include_metadata,
            include_prev_next_rel=include_prev_next_rel,
            id_func=id_func,
        )
        self._tokenizer = tokenizer or get_tokenizer()
        all_seps = [separator] + (backup_separators or [])
        self._split_fns = [split_by_sep(sep) for sep in all_seps] + [split_by_char()]

    @classmethod
    def from_defaults(
        cls,
        chunk_size: int = DEFAULT_CHUNK_SIZE,
        chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
        separator: str = " ",
        backup_separators: list[str] | None = ["\n"],
        callback_manager: CallbackManager | None = None,
        keep_whitespaces: bool = False,
        include_metadata: bool = True,
        include_prev_next_rel: bool = True,
        id_func: Callable[[int, Document], str] | None = None,
    ) -> "TokenMessageSplitter":
        """Initialize with default parameters."""
        callback_manager = callback_manager or CallbackManager([])
        return cls(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separator=separator,
            backup_separators=backup_separators,
            keep_whitespaces=keep_whitespaces,
            callback_manager=callback_manager,
            include_metadata=include_metadata,
            include_prev_next_rel=include_prev_next_rel,
            id_func=id_func,
        )

    @classmethod
    def class_name(cls) -> str:
        return "TokenTextSplitter"

    def split_message_metadata_aware(
        self, message: ChatMessage, metadata_str: str
    ) -> list[ChatMessage]:
        """Split text into chunks, reserving space required for metadata str."""
        metadata_len = len(self._tokenizer(metadata_str)) + DEFAULT_METADATA_FORMAT_LEN
        effective_chunk_size = self.chunk_size - metadata_len
        if effective_chunk_size <= 0:
            raise ValueError(
                f"Metadata length ({metadata_len}) is longer than chunk size "
                f"({self.chunk_size}). Consider increasing the chunk size or "
                "decreasing the size of your metadata to avoid this."
            )
        elif effective_chunk_size < 50:
            print(
                f"Metadata length ({metadata_len}) is close to chunk size "
                f"({self.chunk_size}). Resulting chunks are less than 50 tokens. "
                "Consider increasing the chunk size or decreasing the size of "
                "your metadata to avoid this.",
                flush=True,
            )

        return self._split_message(message, chunk_size=effective_chunk_size)

    def split_message(self, message: ChatMessage) -> list[ChatMessage]:
        return self._split_message(message, chunk_size=self.chunk_size)

    def _split_message(
        self, message: ChatMessage, chunk_size: int
    ) -> list[ChatMessage]:
        if message.blocks == []:
            return [message]

        with self.callback_manager.event(
            CBEventType.CHUNKING, payload={EventPayload.CHUNKS: [message]}
        ) as event:
            splits = self._split(message, chunk_size)
            chunks = self._merge(splits, chunk_size)

            event.on_end(
                payload={EventPayload.CHUNKS: chunks},
            )

        return chunks

    def _split(self, message: ChatMessage, chunk_size: int) -> list[ChatMessage]:
        """
        Break messages into splits that are smaller than chunk size.

        The order of splitting is:
        1. split by separator
        2. split by backup separators (if any)
        3. split by characters

        NOTE: the splits contain the separators.
        """
        splits = []
        for block in message.blocks:
            # Only split text blocks
            if isinstance(block, TextBlock):
                if len(self._tokenizer(block.text)) < chunk_size:
                    splits.append(ChatMessage(blocks=[block]))
                else:
                    for split_fn in self._split_fns:
                        sub_splits = split_fn(block)
                        if len(sub_splits) > 1:
                            splits += [
                                ChatMessage(
                                    blocks=[
                                        TextBlock(text=sub_split)
                                        for sub_split in sub_splits
                                    ]
                                )
                            ]
                            break
                    # What happens if a message can't be split? here it seems like it just gets cut
                    # (although in practice it should never reach here because of the char splitter)
                    # Alternatively, adding it back in unsplit would cause an infinite loop
                    # Probably, I deally this should throw an explicit error instead
            else:
                splits.append(ChatMessage(blocks=[block]))

        new_splits = []
        for split_message in splits:
            # At this point every message has one block
            # Only deal with text blocks, but pass other blocks through
            if isinstance(block := split_message.blocks[0], TextBlock):
                split_len = len(self._tokenizer(block.text))
                if split_len <= chunk_size:
                    new_splits.append(split_message)
                else:
                    # recursively split
                    new_splits.extend(self._split(split_message, chunk_size=chunk_size))
            else:
                new_splits.append(split_message)
        return new_splits

    def _merge(self, splits: list[ChatMessage], chunk_size: int) -> list[ChatMessage]:
        """
        Merge splits into chunks.

        The high-level idea is to keep adding splits to a chunk until we
        exceed the chunk size, then we start a new chunk with overlap.

        When we start a new chunk, we pop off the first element of the previous
        chunk until the total length is less than the chunk size.
        """
        messages: list[ChatMessage] = []

        cur_message: ChatMessage = ChatMessage(blocks=[])
        cur_len = 0

        # At some point it might back sense to move token counting to the block types themselves so you could just call
        # block.get_token_length(self._tokenizer)
        for split in splits:
            block = split.blocks[0]
            split_len = block.estimate_tokens(tokenizer=self._tokenizer)

            if split_len > chunk_size:
                _logger.warning(
                    f"Got a split of size {split_len}, ",
                    f"larger than chunk size {chunk_size}.",
                )

            # if we exceed the chunk size after adding the new split, then
            # we need to end the current chunk and start a new one
            if cur_len + split_len > chunk_size:
                # end the previous chunk
                messages.append(cur_message)
                cur_message = ChatMessage(blocks=[])
                cur_len = 0

                # # start a new chunk with overlap
                # # keep popping off the first element of the previous chunk until:
                # #   1. the current chunk length is less than chunk overlap
                # #   2. the total length is less than chunk size
                # while cur_len > self.chunk_overlap or cur_len + split_len > chunk_size:
                #     # pop off the first element
                #     first_chunk = cur_chunk.pop(0)
                #     cur_len -= len(self._tokenizer(first_chunk))

            cur_message.blocks.append(block)
            cur_len += split_len

        # handle the last chunk
        messages.append(cur_message)

        return messages
