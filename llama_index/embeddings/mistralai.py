"""MistralAI embeddings file."""

from typing import Any, List, Optional

from llama_index.bridge.pydantic import Field
from llama_index.callbacks.base import CallbackManager
from llama_index.embeddings.base import DEFAULT_EMBED_BATCH_SIZE, BaseEmbedding


class MistralAIEmbedding(BaseEmbedding):
    """Class for MistralAI embeddings.

    Args:
        model_name (str): Model for embedding.
            Defaults to "mistral-embed".

        api_key (Optional[str]): API key to access the model. Defaults to None.
    """

    # Instance variables initialized via Pydantic's mechanism
    mistralai_client: Any = Field(description="MistralAI client")

    def __init__(
        self,
        model_name: str = "mistral-embed",
        api_key: Optional[str] = None,
        embed_batch_size: int = DEFAULT_EMBED_BATCH_SIZE,
        callback_manager: Optional[CallbackManager] = None,
        **kwargs: Any,
    ):
        try:
            from mistralai.client import MistralClient
        except ImportError:
            raise ImportError(
                "mistralai package not found, install with" "'pip install mistralai'"
            )
        super().__init__(
            mistralai_client=MistralClient(api_key=api_key),
            model_name=model_name,
            embed_batch_size=embed_batch_size,
            callback_manager=callback_manager,
            **kwargs,
        )

    @classmethod
    def class_name(cls) -> str:
        return "MistralAIEmbedding"

    def _get_query_embedding(self, query: str) -> List[float]:
        """Get query embedding."""
        return (
            self.mistralai_client.embeddings(model=self.model_name, input=[query])
            .data[0]
            .embedding
        )

    async def _aget_query_embedding(self, query: str) -> List[float]:
        """The asynchronous version of _get_query_embedding."""
        return await self._get_query_embedding(query)

    def _get_text_embedding(self, text: str) -> List[float]:
        """Get text embedding."""
        return (
            self.mistralai_client.embeddings(model=self.model_name, input=[text])
            .data[0]
            .embedding
        )

    async def _aget_text_embedding(self, text: str) -> List[float]:
        """Asynchronously get text embedding."""
        return self._get_text_embedding(text)

    def _get_text_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Get text embeddings."""
        embedding_response = self.mistralai_client.embeddings(
            model=self.model_name, input=texts
        ).data
        return [embed.embedding for embed in embedding_response]

    async def _aget_text_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Asynchronously get text embeddings."""
        return await self._get_text_embeddings(texts)
