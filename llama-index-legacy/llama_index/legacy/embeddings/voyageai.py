"""Voyage embeddings file."""

from typing import Any, List, Optional

from llama_index.legacy.bridge.pydantic import PrivateAttr
from llama_index.legacy.callbacks.base import CallbackManager
from llama_index.legacy.embeddings.base import BaseEmbedding

DEFAULT_VOYAGE_BATCH_SIZE = 8


class VoyageEmbedding(BaseEmbedding):
    """Class for Voyage embeddings.

    Args:
        model_name (str): Model for embedding.
            Defaults to "voyage-01".

        voyage_api_key (Optional[str]): Voyage API key. Defaults to None.
            You can either specify the key here or store it as an environment variable.
    """

    _model: Any = PrivateAttr()

    def __init__(
        self,
        model_name: str = "voyage-01",
        voyage_api_key: Optional[str] = None,
        embed_batch_size: int = DEFAULT_VOYAGE_BATCH_SIZE,
        callback_manager: Optional[CallbackManager] = None,
        **kwargs: Any,
    ):
        try:
            import voyageai
        except ImportError:
            raise ImportError(
                "voyageai package not found, install with" "'pip install voyageai'"
            )
        if voyage_api_key:
            voyageai.api_key = voyage_api_key
        self._model = voyageai

        super().__init__(
            model_name=model_name,
            embed_batch_size=embed_batch_size,
            callback_manager=callback_manager,
            **kwargs,
        )

    @classmethod
    def class_name(cls) -> str:
        return "VoyageEmbedding"

    def _get_query_embedding(self, query: str) -> List[float]:
        """Get query embedding."""
        return self._model.get_embedding(
            query, model=self.model_name, input_type="query"
        )

    async def _aget_query_embedding(self, query: str) -> List[float]:
        """The asynchronous version of _get_query_embedding."""
        return await self._model.aget_embedding(
            query, model=self.model_name, input_type="query"
        )

    def _get_text_embedding(self, text: str) -> List[float]:
        """Get text embedding."""
        return self._model.get_embedding(
            text, model=self.model_name, input_type="document"
        )

    async def _aget_text_embedding(self, text: str) -> List[float]:
        """Asynchronously get text embedding."""
        return await self._model.aget_embedding(
            text, model=self.model_name, input_type="document"
        )

    def _get_text_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Get text embeddings."""
        return self._model.get_embeddings(
            texts, model=self.model_name, input_type="document"
        )

    async def _aget_text_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Asynchronously get text embeddings."""
        return await self._model.aget_embeddings(
            texts, model=self.model_name, input_type="document"
        )

    def get_general_text_embedding(
        self, text: str, input_type: Optional[str] = None
    ) -> List[float]:
        """Get general text embedding with input_type."""
        return self._model.get_embedding(
            text, model=self.model_name, input_type=input_type
        )

    async def aget_general_text_embedding(
        self, text: str, input_type: Optional[str] = None
    ) -> List[float]:
        """Asynchronously get general text embedding with input_type."""
        return await self._model.aget_embedding(
            text, model=self.model_name, input_type=input_type
        )
