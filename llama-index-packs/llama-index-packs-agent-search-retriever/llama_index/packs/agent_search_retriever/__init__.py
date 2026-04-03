import warnings

warnings.warn(
    "llama-index-packs-agent-search-retriever is deprecated and no longer maintained. "
    "It will not receive any further updates.",
    DeprecationWarning,
    stacklevel=2,
)

from llama_index.packs.agent_search_retriever.base import (
    AgentSearchRetriever,
    AgentSearchRetrieverPack,
)

__all__ = ["AgentSearchRetriever", "AgentSearchRetrieverPack"]
