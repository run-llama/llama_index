import warnings

warnings.warn(
    "llama-index-packs-searchain is deprecated and no longer maintained. "
    "It will not receive any further updates.",
    DeprecationWarning,
    stacklevel=2,
)

from llama_index.packs.searchain.base import SearChainPack


__all__ = ["SearChainPack"]
