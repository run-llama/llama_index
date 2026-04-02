import warnings

warnings.warn(
    "llama-index-packs-gmail-openai-agent is deprecated and no longer maintained. "
    "It will not receive any further updates.",
    DeprecationWarning,
    stacklevel=2,
)

from llama_index.packs.gmail_openai_agent.base import GmailOpenAIAgentPack

__all__ = ["GmailOpenAIAgentPack"]
