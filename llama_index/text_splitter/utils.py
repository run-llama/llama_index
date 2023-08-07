from typing import Optional

from llama_index.callbacks.base import CallbackManager
from llama_index.constants import DEFAULT_CHUNK_OVERLAP, DEFAULT_CHUNK_SIZE
from llama_index.text_splitter.sentence_splitter import SentenceSplitter
from llama_index.text_splitter.types import TextSplitter


def truncate_text(text: str, text_splitter: TextSplitter) -> str:
    """Truncate text to fit within the chunk size."""
    chunks = text_splitter.split_text(text)
    return chunks[0]


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


def split_text_keep_separator(text: str, separator: str):
    """Split text with separator and keep the separator at the end of each split."""
    parts = text.split(separator)
    result = [separator + s if i > 0 else s for i, s in enumerate(parts)]
    return result
