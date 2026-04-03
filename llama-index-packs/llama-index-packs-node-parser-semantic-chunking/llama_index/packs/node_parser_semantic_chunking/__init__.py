import warnings

warnings.warn(
    "llama-index-packs-node-parser-semantic-chunking is deprecated and no longer maintained. "
    "It will not receive any further updates.",
    DeprecationWarning,
    stacklevel=2,
)

from llama_index.packs.node_parser_semantic_chunking.base import (
    SemanticChunkingQueryEnginePack,
)

__all__ = ["SemanticChunkingQueryEnginePack"]
