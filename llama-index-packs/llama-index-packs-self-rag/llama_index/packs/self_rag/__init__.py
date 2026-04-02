import warnings

warnings.warn(
    "llama-index-packs-self-rag is deprecated and no longer maintained. "
    "It will not receive any further updates.",
    DeprecationWarning,
    stacklevel=2,
)

from llama_index.packs.self_rag.base import SelfRAGQueryEngine, SelfRAGPack


__all__ = ["SelfRAGPack", "SelfRAGQueryEngine"]
