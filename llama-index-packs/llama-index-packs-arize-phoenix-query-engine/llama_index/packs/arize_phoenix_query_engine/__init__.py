import warnings

warnings.warn(
    "llama-index-packs-arize-phoenix-query-engine is deprecated and no longer maintained. "
    "It will not receive any further updates.",
    DeprecationWarning,
    stacklevel=2,
)

from llama_index.packs.arize_phoenix_query_engine.base import (
    ArizePhoenixQueryEnginePack,
)

__all__ = ["ArizePhoenixQueryEnginePack"]
