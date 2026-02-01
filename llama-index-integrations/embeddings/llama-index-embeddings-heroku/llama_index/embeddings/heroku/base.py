"""Heroku Inference API embedding provider for LlamaIndex."""

from typing import Any, List, Optional

import httpx
from llama_index.core.base.embeddings.base import BaseEmbedding
from llama_index.core.bridge.pydantic import Field, PrivateAttr
from llama_index.core.callbacks.base import CallbackManager


class HerokuEmbedding(BaseEmbedding):
    """Heroku Inference API embedding provider.

    This class wraps the Heroku Inference API's /v1/embeddings endpoint,
    which provides access to Cohere embedding models.

    Args:
        api_key: Your Heroku Inference API key.
        model: The embedding model to use (default: "cohere-embed-multilingual").
        base_url: The Heroku Inference API base URL.
        embed_batch_size: Maximum batch size for embedding requests.
        timeout: Request timeout in seconds.

    Example:
        >>> from llama_index.embeddings.heroku import HerokuEmbedding
        >>> embed = HerokuEmbedding(api_key="your-key")
        >>> vector = embed.get_text_embedding("Hello world")
    """

    api_key: str = Field(description="Heroku Inference API key")
    base_url: str = Field(
        default="https://us.inference.heroku.com",
        description="Heroku Inference API base URL",
    )
    model: str = Field(
        default="cohere-embed-multilingual",
        description="The embedding model to use",
    )
    timeout: float = Field(
        default=60.0,
        description="Request timeout in seconds",
    )
    input_type: Optional[str] = Field(
        default=None,
        description="Input type for the embedding model (e.g., 'search_document', 'search_query')",
    )

    _client: httpx.Client = PrivateAttr()
    _async_client: httpx.AsyncClient = PrivateAttr()

    def __init__(
        self,
        api_key: str,
        model: str = "cohere-embed-multilingual",
        base_url: str = "https://us.inference.heroku.com",
        embed_batch_size: int = 96,
        timeout: float = 60.0,
        input_type: Optional[str] = None,
        callback_manager: Optional[CallbackManager] = None,
        **kwargs: Any,
    ) -> None:
        """Initialize the Heroku embedding provider."""
        super().__init__(
            api_key=api_key,
            model=model,
            base_url=base_url,
            embed_batch_size=embed_batch_size,
            timeout=timeout,
            input_type=input_type,
            callback_manager=callback_manager,
            **kwargs,
        )
        self._client = httpx.Client(timeout=timeout)
        self._async_client = httpx.AsyncClient(timeout=timeout)

    def _get_headers(self) -> dict:
        """Get request headers."""
        return {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

    def _get_request_body(self, texts: List[str]) -> dict:
        """Build the request body for the embeddings API."""
        body = {
            "model": self.model,
            "input": texts,
        }
        if self.input_type:
            body["input_type"] = self.input_type
        return body

    def _get_text_embedding(self, text: str) -> List[float]:
        """Get embedding for a single text.

        Args:
            text: The text to embed.

        Returns:
            The embedding vector as a list of floats.
        """
        response = self._client.post(
            f"{self.base_url}/v1/embeddings",
            headers=self._get_headers(),
            json=self._get_request_body([text]),
        )
        response.raise_for_status()
        return response.json()["data"][0]["embedding"]

    def _get_text_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Get embeddings for multiple texts.

        Args:
            texts: The texts to embed.

        Returns:
            A list of embedding vectors.
        """
        response = self._client.post(
            f"{self.base_url}/v1/embeddings",
            headers=self._get_headers(),
            json=self._get_request_body(texts),
        )
        response.raise_for_status()
        data = response.json()["data"]
        # Sort by index to ensure correct order
        sorted_data = sorted(data, key=lambda x: x["index"])
        return [d["embedding"] for d in sorted_data]

    async def _aget_text_embedding(self, text: str) -> List[float]:
        """Async get embedding for a single text.

        Args:
            text: The text to embed.

        Returns:
            The embedding vector as a list of floats.
        """
        response = await self._async_client.post(
            f"{self.base_url}/v1/embeddings",
            headers=self._get_headers(),
            json=self._get_request_body([text]),
        )
        response.raise_for_status()
        return response.json()["data"][0]["embedding"]

    async def _aget_text_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Async get embeddings for multiple texts.

        Args:
            texts: The texts to embed.

        Returns:
            A list of embedding vectors.
        """
        response = await self._async_client.post(
            f"{self.base_url}/v1/embeddings",
            headers=self._get_headers(),
            json=self._get_request_body(texts),
        )
        response.raise_for_status()
        data = response.json()["data"]
        # Sort by index to ensure correct order
        sorted_data = sorted(data, key=lambda x: x["index"])
        return [d["embedding"] for d in sorted_data]

    def _get_query_embedding(self, query: str) -> List[float]:
        """Get embedding for a query.

        For Cohere models, queries should use input_type='search_query'.

        Args:
            query: The query text to embed.

        Returns:
            The embedding vector as a list of floats.
        """
        # Temporarily set input_type for query embedding
        body = self._get_request_body([query])
        if self.model.startswith("cohere"):
            body["input_type"] = "search_query"

        response = self._client.post(
            f"{self.base_url}/v1/embeddings",
            headers=self._get_headers(),
            json=body,
        )
        response.raise_for_status()
        return response.json()["data"][0]["embedding"]

    async def _aget_query_embedding(self, query: str) -> List[float]:
        """Async get embedding for a query.

        For Cohere models, queries should use input_type='search_query'.

        Args:
            query: The query text to embed.

        Returns:
            The embedding vector as a list of floats.
        """
        body = self._get_request_body([query])
        if self.model.startswith("cohere"):
            body["input_type"] = "search_query"

        response = await self._async_client.post(
            f"{self.base_url}/v1/embeddings",
            headers=self._get_headers(),
            json=body,
        )
        response.raise_for_status()
        return response.json()["data"][0]["embedding"]

    @classmethod
    def class_name(cls) -> str:
        """Return the class name."""
        return "HerokuEmbedding"
