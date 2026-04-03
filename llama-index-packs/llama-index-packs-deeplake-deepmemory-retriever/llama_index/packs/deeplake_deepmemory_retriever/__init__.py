import warnings

warnings.warn(
    "llama-index-packs-deeplake-deepmemory-retriever is deprecated and no longer maintained. "
    "It will not receive any further updates.",
    DeprecationWarning,
    stacklevel=2,
)

from llama_index.packs.deeplake_deepmemory_retriever.base import DeepMemoryRetrieverPack

__all__ = ["DeepMemoryRetrieverPack"]
