import warnings

warnings.warn(
    "llama-index-packs-resume-screener is deprecated and no longer maintained. "
    "It will not receive any further updates.",
    DeprecationWarning,
    stacklevel=2,
)

from llama_index.packs.resume_screener.base import ResumeScreenerPack

__all__ = ["ResumeScreenerPack"]
