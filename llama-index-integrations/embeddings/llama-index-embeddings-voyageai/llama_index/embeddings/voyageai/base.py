"""Voyage embeddings file."""

import logging
from typing import Any, List, Optional

from llama_index.core.bridge.pydantic import PrivateAttr
from llama_index.core.base.embeddings.base import BaseEmbedding
from llama_index.core.callbacks.base import CallbackManager

import voyageai

logger = logging.getLogger(__name__)


class VoyageEmbedding(BaseEmbedding):
    """Class for Voyage embeddings.

    Args:
        model_name (str): Model for embedding.
            Defaults to "voyage-01".

        voyage_api_key (Optional[str]): Voyage API key. Defaults to None.
            You can either specify the key here or store it as an environment variable.
    """

    _client: voyageai.Client = PrivateAttr(None)
    _aclient: voyageai.client_async.AsyncClient = PrivateAttr()
    truncation: Optional[bool] = None

    def __init__(
        self,
        model_name: str,
        voyage_api_key: Optional[str] = None,
        embed_batch_size: Optional[int] = None,
        truncation: Optional[bool] = None,
        callback_manager: Optional[CallbackManager] = None,
        **kwargs: Any,
    ):
        if model_name in [
            "voyage-01",
            "voyage-lite-01",
            "voyage-lite-01-instruct",
            "voyage-02",
            "voyage-2",
            "voyage-lite-02-instruct",
        ]:
            logger.warning(
                f"{model_name} is not the latest model by Voyage AI. Please note that `model_name` "
                "will be a required argument in the future. We recommend setting it explicitly. Please see "
                "https://docs.voyageai.com/docs/embeddings for the latest models offered by Voyage AI."
            )

        if embed_batch_size is None:
            embed_batch_size = 72 if model_name in ["voyage-2", "voyage-02"] else 7

        super().__init__(
            model_name=model_name,
            embed_batch_size=embed_batch_size,
            callback_manager=callback_manager,
            **kwargs,
        )

        self._client = voyageai.Client(api_key=voyage_api_key)
        self._aclient = voyageai.AsyncClient(api_key=voyage_api_key)
        self.truncation = truncation

    @classmethod
    def class_name(cls) -> str:
        return "VoyageEmbedding"

    def _get_embedding(self, texts: List[str], input_type: str) -> List[List[float]]:
        return self._client.embed(
            texts,
            model=self.model_name,
            input_type=input_type,
            truncation=self.truncation,
        ).embeddings

    async def _aget_embedding(
        self, texts: List[str], input_type: str
    ) -> List[List[float]]:
        r = await self._aclient.embed(
            texts,
            model=self.model_name,
            input_type=input_type,
            truncation=self.truncation,
        )
        return r.embeddings

    def _get_query_embedding(self, query: str) -> List[float]:
        """Get query embedding."""
        return self._get_embedding([query], input_type="query")[0]

    async def _aget_query_embedding(self, query: str) -> List[float]:
        """The asynchronous version of _get_query_embedding."""
        r = await self._aget_embedding([query], input_type="query")
        return r[0]

    def _get_text_embedding(self, text: str) -> List[float]:
        """Get text embedding."""
        return self._get_embedding([text], input_type="document")[0]

    async def _aget_text_embedding(self, text: str) -> List[float]:
        """Asynchronously get text embedding."""
        r = await self._aget_embedding([text], input_type="document")
        return r[0]

    def _get_text_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Get text embeddings."""
        return self._get_embedding(texts, input_type="document")

    async def _aget_text_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Asynchronously get text embeddings."""
        return await self._aget_embedding(texts, input_type="document")

    def get_general_text_embedding(
        self, text: str, input_type: Optional[str] = None
    ) -> List[float]:
        """Get general text embedding with input_type."""
        return self._get_embedding([text], input_type=input_type)[0]

    async def aget_general_text_embedding(
        self, text: str, input_type: Optional[str] = None
    ) -> List[float]:
        """Asynchronously get general text embedding with input_type."""
        r = await self._aget_embedding([text], input_type=input_type)
        return r[0]
