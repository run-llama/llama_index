"""Heroku embeddings file."""

import logging
from typing import Any, Optional

import httpx
from llama_index.core.base.embeddings.base import (
    DEFAULT_EMBED_BATCH_SIZE,
    BaseEmbedding,
    Embedding,
)
from llama_index.core.bridge.pydantic import Field, PrivateAttr
from llama_index.core.callbacks.base import CallbackManager
from llama_index.core.base.llms.generic_utils import get_from_param_or_env

logger = logging.getLogger(__name__)

DEFAULT_HEROKU_API_BASE = "https://api.heroku.com"
DEFAULT_HEROKU_API_VERSION = "v1"


class HerokuEmbedding(BaseEmbedding):
    """
    Heroku Managed Inference Embeddings Integration.

    This class provides an interface to Heroku's Managed Inference API for embeddings.
    It connects to your Heroku app's embedding endpoint for embedding models. For more
    information about Heroku's embedding endpoint
    see: https://devcenter.heroku.com/articles/heroku-inference-api-model-cohere-embed-multilingual

    Args:
        model (str, optional): The model to use. If not provided, will use EMBEDDING_MODEL_ID.
        api_key (str, optional): The API key for Heroku embedding. Defaults to EMBEDDING_KEY.
        base_url (str, optional): The base URL for embedding. Defaults to EMBEDDING_URL.
        timeout (float, optional): Timeout for requests in seconds. Defaults to 60.0.
        **kwargs: Additional keyword arguments.

    Environment Variables:
        - EMBEDDING_KEY: The API key for Heroku embedding
        - EMBEDDING_URL: The base URL for embedding endpoint
        - EMBEDDING_MODEL_ID: The model ID to use

    Raises:
        ValueError: If required environment variables are not set.

    """

    model: Optional[str] = Field(
        default=None, description="The model to use for embeddings."
    )
    api_key: Optional[str] = Field(
        default=None, description="The API key for Heroku embedding."
    )
    base_url: Optional[str] = Field(
        default=None, description="The base URL for embedding endpoint."
    )
    timeout: float = Field(default=60.0, description="Timeout for requests in seconds.")

    _client: httpx.Client = PrivateAttr()
    _aclient: httpx.AsyncClient = PrivateAttr()

    def __init__(
        self,
        model: Optional[str] = None,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        timeout: float = 60.0,
        embed_batch_size: int = DEFAULT_EMBED_BATCH_SIZE,
        callback_manager: Optional[CallbackManager] = None,
        **kwargs: Any,
    ) -> None:
        """
        Initialize an instance of the HerokuEmbedding class.

        Args:
            model (str, optional): The model to use. If not provided, will use EMBEDDING_MODEL_ID.
            api_key (str, optional): The API key for Heroku embedding. Defaults to EMBEDDING_KEY.
            base_url (str, optional): The base URL for embedding. Defaults to EMBEDDING_URL.
            timeout (float, optional): Timeout for requests in seconds. Defaults to 60.0.
            embed_batch_size (int, optional): Batch size for embedding calls. Defaults to DEFAULT_EMBED_BATCH_SIZE.
            callback_manager (Optional[CallbackManager], optional): Callback manager. Defaults to None.
            **kwargs: Additional keyword arguments.

        """
        # Get API key from parameter or environment
        try:
            api_key = get_from_param_or_env(
                "api_key",
                api_key,
                "EMBEDDING_KEY",
            )
        except ValueError:
            raise ValueError(
                "API key is required. Set EMBEDDING_KEY environment variable or pass api_key parameter."
            )

        # Get embedding URL from parameter or environment
        try:
            base_url = get_from_param_or_env(
                "base_url",
                base_url,
                "EMBEDDING_URL",
            )
        except ValueError:
            raise ValueError(
                "Embedding URL is required. Set EMBEDDING_URL environment variable or pass base_url parameter."
            )

        # Get model from parameter or environment
        try:
            model = get_from_param_or_env(
                "model",
                model,
                "EMBEDDING_MODEL_ID",
            )
        except ValueError:
            raise ValueError(
                "Model is required. Set EMBEDDING_MODEL_ID environment variable or pass model parameter."
            )

        super().__init__(
            model_name=model,
            embed_batch_size=embed_batch_size,
            callback_manager=callback_manager,
            **kwargs,
        )

        self.model = model
        self.api_key = api_key
        self.base_url = base_url
        self.timeout = timeout

        # Initialize HTTP clients
        self._client = httpx.Client(
            timeout=timeout,
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
                "User-Agent": "llama-index-embeddings-heroku",
            },
        )
        self._aclient = httpx.AsyncClient(
            timeout=timeout,
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
                "User-Agent": "llama-index-embeddings-heroku",
            },
        )

    @classmethod
    def class_name(cls) -> str:
        """Get class name."""
        return "HerokuEmbedding"

    def _get_query_embedding(self, query: str) -> Embedding:
        """Get query embedding."""
        return self._get_text_embedding(query)

    def _get_text_embedding(self, text: str) -> Embedding:
        """Get text embedding."""
        try:
            response = self._client.post(
                f"{self.base_url}/v1/embeddings",
                json={
                    "input": text,
                    "model": self.model,
                },
            )
            response.raise_for_status()
            data = response.json()
            return data["data"][0]["embedding"]
        except httpx.HTTPStatusError as e:
            logger.error(f"HTTP error while embedding text: {e}")
            raise ValueError(f"Unable to embed text: {e}")
        except Exception as e:
            logger.error(f"Error while embedding text: {e}")
            raise ValueError(f"Unable to embed text: {e}")

    async def _aget_query_embedding(self, query: str) -> Embedding:
        """Get query embedding asynchronously."""
        return await self._aget_text_embedding(query)

    async def _aget_text_embedding(self, text: str) -> Embedding:
        """Get text embedding asynchronously."""
        try:
            response = await self._aclient.post(
                f"{self.base_url}/v1/embeddings",
                json={
                    "input": text,
                    "model": self.model,
                },
            )
            response.raise_for_status()
            data = response.json()
            return data["data"][0]["embedding"]
        except httpx.HTTPStatusError as e:
            logger.error(f"HTTP error while embedding text: {e}")
            raise ValueError(f"Unable to embed text: {e}")
        except Exception as e:
            logger.error(f"Error while embedding text: {e}")
            raise ValueError(f"Unable to embed text: {e}")

    def __del__(self) -> None:
        """Clean up resources."""
        if hasattr(self, "_client"):
            self._client.close()

    async def aclose(self) -> None:
        """Close async client."""
        if hasattr(self, "_aclient"):
            await self._aclient.aclose()
