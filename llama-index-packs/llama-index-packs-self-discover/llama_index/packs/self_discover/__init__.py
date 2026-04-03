import warnings

warnings.warn(
    "llama-index-packs-self-discover is deprecated and no longer maintained. "
    "It will not receive any further updates.",
    DeprecationWarning,
    stacklevel=2,
)

from llama_index.packs.self_discover.base import SelfDiscoverPack

__all__ = ["SelfDiscoverPack"]
