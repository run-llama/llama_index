"""KG-based data structures."""

from llama_index.indices.knowledge_graph.base import (
    GPTKnowledgeGraphIndex,
    KnowledgeGraphIndex,
)
from llama_index.indices.knowledge_graph.retriever import KGTableRetriever

__all__ = [
    "KnowledgeGraphIndex",
    "KGTableRetriever",
    # legacy
    "GPTKnowledgeGraphIndex",
]
