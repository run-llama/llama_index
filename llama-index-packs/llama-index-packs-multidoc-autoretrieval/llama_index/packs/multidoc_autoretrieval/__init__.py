import warnings

warnings.warn(
    "llama-index-packs-multidoc-autoretrieval is deprecated and no longer maintained. "
    "It will not receive any further updates.",
    DeprecationWarning,
    stacklevel=2,
)

from llama_index.packs.multidoc_autoretrieval.base import MultiDocAutoRetrieverPack

__all__ = ["MultiDocAutoRetrieverPack"]
