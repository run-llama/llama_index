import warnings

warnings.warn(
    "llama-index-packs-dense-x-retrieval is deprecated and no longer maintained. "
    "It will not receive any further updates.",
    DeprecationWarning,
    stacklevel=2,
)

from llama_index.packs.dense_x_retrieval.base import DenseXRetrievalPack

__all__ = ["DenseXRetrievalPack"]
