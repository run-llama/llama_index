"""MistralAI embeddings file."""

from typing import Any, List, Optional

from llama_index.core.base.embeddings.base import (
    DEFAULT_EMBED_BATCH_SIZE,
    BaseEmbedding,
)
from llama_index.core.bridge.pydantic import PrivateAttr
from llama_index.core.callbacks.base import CallbackManager
from llama_index.core.base.llms.generic_utils import get_from_param_or_env

from mistralai import Mistral


class MistralAIEmbedding(BaseEmbedding):
    """
    Class for MistralAI embeddings.

    Args:
        model_name (str): Model for embedding.
            Defaults to "mistral-embed".

        api_key (Optional[str]): API key to access the model. Defaults to None.

    """

    # Instance variables initialized via Pydantic's mechanism
    _client: Mistral = PrivateAttr()

    def __init__(
        self,
        model_name: str = "mistral-embed",
        api_key: Optional[str] = None,
        embed_batch_size: int = DEFAULT_EMBED_BATCH_SIZE,
        callback_manager: Optional[CallbackManager] = None,
        **kwargs: Any,
    ):
        super().__init__(
            model_name=model_name,
            embed_batch_size=embed_batch_size,
            callback_manager=callback_manager,
            **kwargs,
        )
        api_key = get_from_param_or_env("api_key", api_key, "MISTRAL_API_KEY", "")

        if not api_key:
            raise ValueError(
                "You must provide an API key to use mistralai. "
                "You can either pass it in as an argument or set it `MISTRAL_API_KEY`."
            )
        self._client = Mistral(api_key=api_key)

    @classmethod
    def class_name(cls) -> str:
        return "MistralAIEmbedding"

    def _get_query_embedding(self, query: str) -> List[float]:
        """Get query embedding."""
        return (
            self._client.embeddings.create(model=self.model_name, inputs=[query])
            .data[0]
            .embedding
        )

    async def _aget_query_embedding(self, query: str) -> List[float]:
        """The asynchronous version of _get_query_embedding."""
        return (
            (
                await self._client.embeddings.create_async(
                    model=self.model_name, inputs=[query]
                )
            )
            .data[0]
            .embedding
        )

    def _get_text_embedding(self, text: str) -> List[float]:
        """Get text embedding."""
        return (
            self._client.embeddings.create(model=self.model_name, inputs=[text])
            .data[0]
            .embedding
        )

    async def _aget_text_embedding(self, text: str) -> List[float]:
        """Asynchronously get text embedding."""
        return (
            await self._client.embeddings.create(
                model=self.model_name,
                inputs=[text],
            )
            .data[0]
            .embedding
        )

    def _get_text_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Get text embeddings."""
        embedding_response = self._client.embeddings.create(
            model=self.model_name, inputs=texts
        ).data
        return [embed.embedding for embed in embedding_response]

    async def _aget_text_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Asynchronously get text embeddings."""
        embedding_response = await self._client.embeddings.create_async(
            model=self.model_name, inputs=texts
        )
        return [embed.embedding for embed in embedding_response.data]
