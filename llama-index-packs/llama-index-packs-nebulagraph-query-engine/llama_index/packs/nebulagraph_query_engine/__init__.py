import warnings

warnings.warn(
    "llama-index-packs-nebulagraph-query-engine is deprecated and no longer maintained. "
    "It will not receive any further updates.",
    DeprecationWarning,
    stacklevel=2,
)

from llama_index.packs.nebulagraph_query_engine.base import NebulaGraphQueryEnginePack

__all__ = ["NebulaGraphQueryEnginePack"]
