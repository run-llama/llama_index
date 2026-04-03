import warnings

warnings.warn(
    "llama-index-packs-mixture-of-agents is deprecated and no longer maintained. "
    "It will not receive any further updates.",
    DeprecationWarning,
    stacklevel=2,
)

from llama_index.packs.mixture_of_agents.base import MixtureOfAgentsPack

__all__ = ["MixtureOfAgentsPack"]
