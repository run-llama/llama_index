import warnings

warnings.warn(
    "llama-index-packs-snowflake-query-engine is deprecated and no longer maintained. "
    "It will not receive any further updates.",
    DeprecationWarning,
    stacklevel=2,
)

from llama_index.packs.snowflake_query_engine.base import SnowflakeQueryEnginePack

__all__ = ["SnowflakeQueryEnginePack"]
