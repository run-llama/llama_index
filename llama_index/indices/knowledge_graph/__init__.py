"""KG-based data structures."""

from llama_index.indices.knowledge_graph.base import (
    GPTKnowledgeGraphIndex,
    KnowledgeGraphIndex,
)
from llama_index.indices.knowledge_graph.retrievers import (
    KGTableRetriever,
    KnowledgeGraphRAGRetriever,
)

__all__ = [
    "KnowledgeGraphIndex",
    "KGTableRetriever",
    "KnowledgeGraphRAGRetriever",
    # legacy
    "GPTKnowledgeGraphIndex",
]
