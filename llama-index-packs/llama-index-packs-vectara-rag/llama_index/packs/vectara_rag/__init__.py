import warnings

warnings.warn(
    "llama-index-packs-vectara-rag is deprecated and no longer maintained. "
    "It will not receive any further updates.",
    DeprecationWarning,
    stacklevel=2,
)

from llama_index.packs.vectara_rag.base import VectaraRagPack

__all__ = ["VectaraRagPack"]
