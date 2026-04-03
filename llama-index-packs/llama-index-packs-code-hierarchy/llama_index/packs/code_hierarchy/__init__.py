import warnings

warnings.warn(
    "llama-index-packs-code-hierarchy is deprecated and no longer maintained. "
    "It will not receive any further updates.",
    DeprecationWarning,
    stacklevel=2,
)

from llama_index.packs.code_hierarchy.base import CodeHierarchyAgentPack
from llama_index.packs.code_hierarchy.code_hierarchy import CodeHierarchyNodeParser
from llama_index.packs.code_hierarchy.query_engine import (
    CodeHierarchyKeywordQueryEngine,
)

__all__ = [
    "CodeHierarchyAgentPack",
    "CodeHierarchyNodeParser",
    "CodeHierarchyKeywordQueryEngine",
]
