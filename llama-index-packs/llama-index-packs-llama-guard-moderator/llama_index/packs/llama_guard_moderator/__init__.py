import warnings

warnings.warn(
    "llama-index-packs-llama-guard-moderator is deprecated and no longer maintained. "
    "It will not receive any further updates.",
    DeprecationWarning,
    stacklevel=2,
)

from llama_index.packs.llama_guard_moderator.base import LlamaGuardModeratorPack

__all__ = ["LlamaGuardModeratorPack"]
