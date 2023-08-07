from llama_index.text_splitter.sentence_splitter import SentenceSplitter
from llama_index.text_splitter.types import TextSplitter


def truncate_text(text: str, text_splitter: TextSplitter) -> str:
    """Truncate text to fit within the chunk size."""
    chunks = text_splitter.split_text(text)
    return chunks[0]


def get_default_text_splitter() -> TextSplitter:
    """Get default text splitter."""
    return SentenceSplitter()
