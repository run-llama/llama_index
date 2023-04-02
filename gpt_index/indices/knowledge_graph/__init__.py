"""KG-based data structures."""

from gpt_index.indices.knowledge_graph.base import GPTKnowledgeGraphIndex
from gpt_index.indices.knowledge_graph.query import GPTKGTableQuery, KGQueryMode

__all__ = [
    "GPTKnowledgeGraphIndex",
    "GPTKGTableQuery",
    "KGQueryMode",
]
