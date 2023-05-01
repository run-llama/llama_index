"""KG-based data structures."""

from gpt_index.indices.knowledge_graph.base import GPTKnowledgeGraphIndex
from gpt_index.indices.knowledge_graph.retrievers import KGTableRetriever

__all__ = [
    "GPTKnowledgeGraphIndex",
    "KGTableRetriever",
]
