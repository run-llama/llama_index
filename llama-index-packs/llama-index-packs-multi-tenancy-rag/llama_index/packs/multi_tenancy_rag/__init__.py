import warnings

warnings.warn(
    "llama-index-packs-multi-tenancy-rag is deprecated and no longer maintained. "
    "It will not receive any further updates.",
    DeprecationWarning,
    stacklevel=2,
)

from llama_index.packs.multi_tenancy_rag.base import MultiTenancyRAGPack

__all__ = ["MultiTenancyRAGPack"]
