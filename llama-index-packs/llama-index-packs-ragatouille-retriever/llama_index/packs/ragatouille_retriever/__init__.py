import warnings

warnings.warn(
    "llama-index-packs-ragatouille-retriever is deprecated and no longer maintained. "
    "It will not receive any further updates.",
    DeprecationWarning,
    stacklevel=2,
)

from llama_index.packs.ragatouille_retriever.base import RAGatouilleRetrieverPack

__all__ = ["RAGatouilleRetrieverPack"]
