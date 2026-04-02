import warnings

warnings.warn(
    "llama-index-packs-voyage-query-engine is deprecated and no longer maintained. "
    "It will not receive any further updates.",
    DeprecationWarning,
    stacklevel=2,
)

from llama_index.packs.voyage_query_engine.base import VoyageQueryEnginePack

__all__ = ["VoyageQueryEnginePack"]
