"""Vector-store based data structures."""

from llama_index.indices.multi_modal.base import MultiModalVectorStoreIndex
from llama_index.indices.multi_modal.retriever import MultiModalVectorIndexRetriever

# from llama_index.indices.multi_modal.base_multi_modal_retriever import MultiModalRetriever

__all__ = [
    "MultiModalVectorStoreIndex",
    "MultiModalVectorIndexRetriever",
    # "MultiModalRetriever",
]
