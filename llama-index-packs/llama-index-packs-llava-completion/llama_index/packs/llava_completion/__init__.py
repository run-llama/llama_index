import warnings

warnings.warn(
    "llama-index-packs-llava-completion is deprecated and no longer maintained. "
    "It will not receive any further updates.",
    DeprecationWarning,
    stacklevel=2,
)

from llama_index.packs.llava_completion.base import LlavaCompletionPack

__all__ = ["LlavaCompletionPack"]
