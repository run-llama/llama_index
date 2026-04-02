import warnings

warnings.warn(
    "llama-index-packs-raptor is deprecated and no longer maintained. "
    "It will not receive any further updates.",
    DeprecationWarning,
    stacklevel=2,
)

from llama_index.packs.raptor.base import RaptorPack, RaptorRetriever


__all__ = ["RaptorPack", "RaptorRetriever"]
