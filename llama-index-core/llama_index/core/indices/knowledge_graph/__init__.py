"""KG-based data structures."""

from llama_index.core.indices.knowledge_graph.base import (
    KnowledgeGraphIndex,
)
from llama_index.core.indices.knowledge_graph.retrievers import (
    KGTableRetriever,
    KnowledgeGraphRAGRetriever,
)

GPTKnowledgeGraphIndex = KnowledgeGraphIndex

__all__ = [
    "KnowledgeGraphIndex",
    "KGTableRetriever",
    "KnowledgeGraphRAGRetriever",
    "GPTKnowledgeGraphIndex",
]
