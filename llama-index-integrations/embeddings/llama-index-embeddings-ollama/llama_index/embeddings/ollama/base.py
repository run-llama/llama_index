import asyncio
from typing import Any, Dict, List, Optional

from llama_index.core.base.embeddings.base import BaseEmbedding
from llama_index.core.bridge.pydantic import Field, PrivateAttr
from llama_index.core.callbacks.base import CallbackManager
from llama_index.core.constants import DEFAULT_EMBED_BATCH_SIZE

from ollama import Client, AsyncClient


class OllamaEmbedding(BaseEmbedding):
    """Class for Ollama embeddings."""

    base_url: str = Field(description="Base url the model is hosted by Ollama")
    model_name: str = Field(description="The Ollama model to use.")
    embed_batch_size: int = Field(
        default=DEFAULT_EMBED_BATCH_SIZE,
        description="The batch size for embedding calls.",
        gt=0,
        le=2048,
    )
    ollama_additional_kwargs: Dict[str, Any] = Field(
        default_factory=dict, description="Additional kwargs for the Ollama API."
    )

    _client: Client = PrivateAttr()
    _async_client: AsyncClient = PrivateAttr()

    def __init__(
        self,
        model_name: str,
        base_url: str = "http://localhost:11434",
        embed_batch_size: int = DEFAULT_EMBED_BATCH_SIZE,
        ollama_additional_kwargs: Optional[Dict[str, Any]] = None,
        callback_manager: Optional[CallbackManager] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            model_name=model_name,
            base_url=base_url,
            embed_batch_size=embed_batch_size,
            ollama_additional_kwargs=ollama_additional_kwargs or {},
            callback_manager=callback_manager,
            **kwargs,
        )

        self._client = Client(host=self.base_url)
        self._async_client = AsyncClient(host=self.base_url)

    @classmethod
    def class_name(cls) -> str:
        return "OllamaEmbedding"

    def _get_query_embedding(self, query: str) -> List[float]:
        """Get query embedding."""
        return self.get_general_text_embedding(query)

    async def _aget_query_embedding(self, query: str) -> List[float]:
        """The asynchronous version of _get_query_embedding."""
        return await self.aget_general_text_embedding(query)

    def _get_text_embedding(self, text: str) -> List[float]:
        """Get text embedding."""
        return self.get_general_text_embedding(text)

    async def _aget_text_embedding(self, text: str) -> List[float]:
        """Asynchronously get text embedding."""
        return await self.aget_general_text_embedding(text)

    def _get_text_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Get text embeddings."""
        embeddings_list: List[List[float]] = []
        for text in texts:
            embeddings = self.get_general_text_embedding(text)
            embeddings_list.append(embeddings)

        return embeddings_list

    async def _aget_text_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Asynchronously get text embeddings."""
        return await asyncio.gather(
            *[self.aget_general_text_embedding(text) for text in texts]
        )

    def get_general_text_embedding(self, texts: str) -> List[float]:
        """Get Ollama embedding."""
        result = self._client.embeddings(
            model=self.model_name, prompt=texts, options=self.ollama_additional_kwargs
        )
        return result["embedding"]

    async def aget_general_text_embedding(self, prompt: str) -> List[float]:
        """Asynchronously get Ollama embedding."""
        result = await self._async_client.embeddings(
            model=self.model_name, prompt=prompt, options=self.ollama_additional_kwargs
        )
        return result["embedding"]
