import warnings

warnings.warn(
    "llama-index-packs-auto-merging-retriever is deprecated and no longer maintained. "
    "It will not receive any further updates.",
    DeprecationWarning,
    stacklevel=2,
)

from llama_index.packs.auto_merging_retriever.base import AutoMergingRetrieverPack

__all__ = ["AutoMergingRetrieverPack"]
