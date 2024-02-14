"""Base embeddings file.

Maintain for backwards compatibility.

"""

from llama_index.legacy.core.embeddings.base import (
    DEFAULT_EMBED_BATCH_SIZE,
    BaseEmbedding,
    Embedding,
    SimilarityMode,
    mean_agg,
    similarity,
)

__all__ = [
    "BaseEmbedding",
    "similarity",
    "SimilarityMode",
    "DEFAULT_EMBED_BATCH_SIZE",
    "mean_agg",
    "Embedding",
]
