import warnings

warnings.warn(
    "llama-index-packs-cohere-citation-chat is deprecated and no longer maintained. "
    "It will not receive any further updates.",
    DeprecationWarning,
    stacklevel=2,
)

from llama_index.packs.cohere_citation_chat.base import CohereCitationChatEnginePack

__all__ = ["CohereCitationChatEnginePack"]
