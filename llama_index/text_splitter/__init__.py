from typing import Optional

from llama_index.callbacks.base import CallbackManager
from llama_index.constants import DEFAULT_CHUNK_OVERLAP, DEFAULT_CHUNK_SIZE
from llama_index.text_splitter.code_splitter import CodeSplitter
from llama_index.text_splitter.sentence_splitter import SentenceSplitter
from llama_index.text_splitter.token_splitter import TokenTextSplitter
from llama_index.text_splitter.types import SplitterType, TextSplitter


def get_default_text_splitter(
    chunk_size: Optional[int] = None,
    chunk_overlap: Optional[int] = None,
    callback_manager: Optional[CallbackManager] = None,
) -> TextSplitter:
    """Get default text splitter."""
    chunk_size = chunk_size or DEFAULT_CHUNK_SIZE
    chunk_overlap = (
        chunk_overlap if chunk_overlap is not None else DEFAULT_CHUNK_OVERLAP
    )

    return SentenceSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        callback_manager=callback_manager,
    )


__all__ = [
    "TextSplitter",
    "TokenTextSplitter",
    "SentenceSplitter",
    "CodeSplitter",
    "SplitterType",
]
