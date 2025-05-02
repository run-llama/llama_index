from typing import Dict, Optional

from llama_index.core.base.embeddings.base_sparse import (
    BaseSparseEmbedding,
    SparseEmbedding,
)
from llama_index.core.bridge.pydantic import Field


class MockSparseEmbedding(BaseSparseEmbedding):
    """A mock sparse embedding model for testing."""

    default_embedding: SparseEmbedding = Field(
        default_factory=lambda: {0: 1.0},
        description="The default embedding to return.",
    )

    text_to_embedding: Optional[Dict[str, SparseEmbedding]] = Field(
        default=None,
        description="The mapping of text to embeddings for lookup.",
    )

    @classmethod
    def class_name(self) -> str:
        return "MockSparseEmbedding"

    def _get_embedding(self, text: str) -> SparseEmbedding:
        if self.text_to_embedding is not None:
            return self.text_to_embedding.get(text, self.default_embedding)
        return self.default_embedding

    def _get_text_embedding(self, text: str) -> SparseEmbedding:
        return self._get_embedding(text)

    async def _aget_text_embedding(self, text: str) -> SparseEmbedding:
        return self._get_embedding(text)

    def _get_query_embedding(self, query: str) -> SparseEmbedding:
        return self._get_embedding(query)

    async def _aget_query_embedding(self, query: str) -> SparseEmbedding:
        return self._get_embedding(query)
