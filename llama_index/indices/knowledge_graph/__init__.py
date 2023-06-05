"""KG-based data structures."""

from llama_index.indices.knowledge_graph.base import KnowledgeGraphIndex, GPTKnowledgeGraphIndex
from llama_index.indices.knowledge_graph.retrievers import KGTableRetriever

__all__ = [
    "KnowledgeGraphIndex",
    "KGTableRetriever",
    # legacy
    "GPTKnowledgeGraphIndex"
]
