import warnings

warnings.warn(
    "llama-index-packs-koda-retriever is deprecated and no longer maintained. "
    "It will not receive any further updates.",
    DeprecationWarning,
    stacklevel=2,
)

from llama_index.packs.koda_retriever.base import KodaRetriever
from llama_index.packs.koda_retriever.matrix import AlphaMatrix
from llama_index.packs.koda_retriever.constants import DEFAULT_CATEGORIES

__all__ = ["KodaRetriever", "AlphaMatrix", "DEFAULT_CATEGORIES"]
