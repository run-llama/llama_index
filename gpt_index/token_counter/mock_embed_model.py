"""Mock embedding model."""

from typing import Any, List

from gpt_index.embeddings.base import BaseEmbedding


class MockEmbedding(BaseEmbedding):
    """Mock embedding.

    Used for token prediction.

    Args:
        embed_dim (int): embedding dimension

    """

    def __init__(self, embed_dim: int, *args: Any, **kwargs: Any) -> None:
        """Init params."""
        super().__init__(*args, **kwargs)
        self.embed_dim = embed_dim

    def _get_query_embedding(self, query: str) -> List[float]:
        """Get query embedding."""
        return [0.5] * self.embed_dim

    def _get_text_embedding(self, text: str) -> List[float]:
        """Get text embedding."""
        return [0.5] * self.embed_dim
