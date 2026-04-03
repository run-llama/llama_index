import warnings

warnings.warn(
    "llama-index-packs-fuzzy-citation is deprecated and no longer maintained. "
    "It will not receive any further updates.",
    DeprecationWarning,
    stacklevel=2,
)

from llama_index.packs.fuzzy_citation.base import FuzzyCitationEnginePack

__all__ = ["FuzzyCitationEnginePack"]
