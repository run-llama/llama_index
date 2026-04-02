import warnings

warnings.warn(
    "llama-index-packs-fusion-retriever is deprecated and no longer maintained. "
    "It will not receive any further updates.",
    DeprecationWarning,
    stacklevel=2,
)

from llama_index.packs.fusion_retriever.hybrid_fusion.base import (
    HybridFusionRetrieverPack,
)
from llama_index.packs.fusion_retriever.query_rewrite.base import (
    QueryRewritingRetrieverPack,
)

__all__ = ["HybridFusionRetrieverPack", "QueryRewritingRetrieverPack"]
