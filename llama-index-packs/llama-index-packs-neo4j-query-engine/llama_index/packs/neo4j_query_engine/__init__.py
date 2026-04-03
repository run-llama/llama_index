import warnings

warnings.warn(
    "llama-index-packs-neo4j-query-engine is deprecated and no longer maintained. "
    "It will not receive any further updates.",
    DeprecationWarning,
    stacklevel=2,
)

from llama_index.packs.neo4j_query_engine.base import Neo4jQueryEnginePack

__all__ = ["Neo4jQueryEnginePack"]
