"""KG-based data structures."""

from llama_index.indices.knowledge_graph.base import GPTKnowledgeGraphIndex
from llama_index.indices.knowledge_graph.retrievers import KGTableRetriever

__all__ = [
    "GPTKnowledgeGraphIndex",
    "KGTableRetriever",
]
