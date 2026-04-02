import warnings

warnings.warn(
    "llama-index-packs-timescale-vector-autoretrieval is deprecated and no longer maintained. "
    "It will not receive any further updates.",
    DeprecationWarning,
    stacklevel=2,
)

from llama_index.packs.timescale_vector_autoretrieval.base import (
    TimescaleVectorAutoretrievalPack,
)

__all__ = ["TimescaleVectorAutoretrievalPack"]
