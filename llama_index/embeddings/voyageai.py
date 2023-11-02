"""OpenAI embeddings file."""

from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

import voyageai

from llama_index.embeddings.base import BaseEmbedding


class VoyageEmbedding(BaseEmbedding):
    """Class for Voyage embeddings.

    Args:
        model (str): Model for embedding.
            Defaults to "voyage-01".

        deployment_name (Optional[str]): Optional deployment of model. Defaults to None.
            If this value is not None, mode and model will be ignored.
            Only available for using AzureOpenAI.
    """

    model: str = "voyage-01"
    voyage_api_key: Optional[str] = None

    def __init__(
        self, 
        model: str = "voyage-01",
        voyage_api_key: Optional[str] = None,
    ):
        voyageai.api_key = voyage_api_key
        super().__init__(model=model)

    @classmethod
    def class_name(cls) -> str:
        return "VoyageEmbedding"

    def _get_query_embedding(self, query: str) -> List[float]:
        """Get query embedding."""
        return voyageai.get_embedding(query, model=self.model)

    async def _aget_query_embedding(self, query: str) -> List[float]:
        """The asynchronous version of _get_query_embedding."""
        return await voyageai.aget_embedding(query, model=self.model)

    def _get_text_embedding(self, text: str) -> List[float]:
        """Get text embedding."""
        return voyageai.get_embedding(text, model=self.model)

    async def _aget_text_embedding(self, text: str) -> List[float]:
        """Asynchronously get text embedding."""
        return await voyageai.aget_embedding(text, model=self.model)

    def _get_text_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Get text embeddings."""
        return voyageai.get_embeddings(texts, model=self.model)

    async def _aget_text_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Asynchronously get text embeddings."""
        return await voyageai.aget_embeddings(texts, model=self.model)
