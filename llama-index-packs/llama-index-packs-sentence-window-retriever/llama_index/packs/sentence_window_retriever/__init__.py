import warnings

warnings.warn(
    "llama-index-packs-sentence-window-retriever is deprecated and no longer maintained. "
    "It will not receive any further updates.",
    DeprecationWarning,
    stacklevel=2,
)

from llama_index.packs.sentence_window_retriever.base import SentenceWindowRetrieverPack

__all__ = ["SentenceWindowRetrieverPack"]
