import warnings

warnings.warn(
    "llama-index-packs-ollama-query-engine is deprecated and no longer maintained. "
    "It will not receive any further updates.",
    DeprecationWarning,
    stacklevel=2,
)

from llama_index.packs.ollama_query_engine.base import OllamaQueryEnginePack

__all__ = ["OllamaQueryEnginePack"]
