"""Voyage embeddings file."""

from typing import Any, List, Optional

from llama_index.core.base.embeddings.base import BaseEmbedding
from llama_index.core.bridge.pydantic import PrivateAttr
from llama_index.core.callbacks.base import CallbackManager

import voyageai

DEFAULT_VOYAGE_BATCH_SIZE = 8


class VoyageEmbedding(BaseEmbedding):
    """Class for Voyage embeddings.

    Args:
        model_name (str): Model for embedding.
            Defaults to "voyage-01".

        voyage_api_key (Optional[str]): Voyage API key. Defaults to None.
            You can either specify the key here or store it as an environment variable.
    """

    client: Optional[voyageai.Client]

    def __init__(
        self,
        model_name: str = "voyage-2",
        voyage_api_key: Optional[str] = None,
        embed_batch_size: int = DEFAULT_VOYAGE_BATCH_SIZE,
        callback_manager: Optional[CallbackManager] = None,
        **kwargs: Any,
    ):
        super().__init__(
            model_name=model_name,
            embed_batch_size=embed_batch_size,
            callback_manager=callback_manager,
            **kwargs,
        )
        self.client = voyageai.Client(api_key=voyage_api_key)

    @classmethod
    def class_name(cls) -> str:
        return "VoyageEmbedding"

    def _get_query_embedding(self, query: str) -> List[float]:
        """Get query embedding."""
        return self.client.embed(
            [query], model=self.model_name, input_type="query"
        ).embeddings

    async def _aget_query_embedding(self, query: str) -> List[float]:
        """The asynchronous version of _get_query_embedding."""
        return await self.client.embed(
            [query], model=self.model_name, input_type="query"
        ).embeddings

    def _get_text_embedding(self, text: str) -> List[float]:
        """Get text embedding."""
        return self.client.embed(
            [text], model=self.model_name, input_type="document"
        ).embeddings

    async def _aget_text_embedding(self, text: str) -> List[float]:
        """Asynchronously get text embedding."""
        return await self.client.embed(
            [text], model=self.model_name, input_type="document"
        ).embeddings

    def _get_text_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Get text embeddings."""
        return self.client.embed(
            texts, model=self.model_name, input_type="document"
        ).embeddings

    async def _aget_text_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Asynchronously get text embeddings."""
        return await self.client.embed(
            texts, model=self.model_name, input_type="document"
        ).embeddings

    def get_general_text_embedding(
        self, text: str, input_type: Optional[str] = None
    ) -> List[float]:
        """Get general text embedding with input_type."""
        return self.client.embed(
            [text], model=self.model_name, input_type=input_type
        ).embeddings

    async def aget_general_text_embedding(
        self, text: str, input_type: Optional[str] = None
    ) -> List[float]:
        """Asynchronously get general text embedding with input_type."""
        return await self.client.embed(
            [text], model=self.model_name, input_type=input_type
        ).embeddings
