import warnings

warnings.warn(
    "llama-index-packs-deeplake-multimodal-retrieval is deprecated and no longer maintained. "
    "It will not receive any further updates.",
    DeprecationWarning,
    stacklevel=2,
)

from llama_index.packs.deeplake_multimodal_retrieval.base import (
    DeepLakeMultimodalRetrieverPack,
)

__all__ = ["DeepLakeMultimodalRetrieverPack"]
