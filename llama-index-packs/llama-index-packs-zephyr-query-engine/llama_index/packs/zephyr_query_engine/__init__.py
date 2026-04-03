import warnings

warnings.warn(
    "llama-index-packs-zephyr-query-engine is deprecated and no longer maintained. "
    "It will not receive any further updates.",
    DeprecationWarning,
    stacklevel=2,
)

from llama_index.packs.zephyr_query_engine.base import ZephyrQueryEnginePack

__all__ = ["ZephyrQueryEnginePack"]
