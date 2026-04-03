import warnings

warnings.warn(
    "llama-index-packs-retry-engine-weaviate is deprecated and no longer maintained. "
    "It will not receive any further updates.",
    DeprecationWarning,
    stacklevel=2,
)

from llama_index.packs.retry_engine_weaviate.base import WeaviateRetryEnginePack

__all__ = ["WeaviateRetryEnginePack"]
