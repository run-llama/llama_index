import warnings

warnings.warn(
    "llama-index-packs-longrag is deprecated and no longer maintained. "
    "It will not receive any further updates.",
    DeprecationWarning,
    stacklevel=2,
)

from llama_index.packs.longrag.base import LongRAGPack


__all__ = ["LongRAGPack"]
