"""Databricks embeddings file."""

from typing import Optional, Any, Dict, List
import httpx
from llama_index.core.bridge.pydantic import Field, PrivateAttr
from llama_index.core.callbacks.base import CallbackManager
from llama_index.embeddings.openai.base import (
    get_embedding,
    aget_embedding,
    get_embeddings,
    aget_embeddings,
)
from llama_index.core.base.embeddings.base import BaseEmbedding
from llama_index.core.base.llms.generic_utils import get_from_param_or_env
from openai import AsyncOpenAI, OpenAI
import logging


logger = logging.getLogger(__name__)


class DatabricksEmbedding(BaseEmbedding):
    """Databricks class for text embedding.

    Databricks adheres to the OpenAI API, so this integration aligns closely with the existing OpenAIEmbedding class.

    Args:
        model (str): The unique ID of the embedding model as served by the Databricks endpoint.
        endpoint (Optional[str]): The url of the Databricks endpoint. Can be set as an environment variable (`DATABRICKS_SERVING_ENDPOINT`).
        api_key (Optional[str]): The Databricks API key to use. Can be set as an environment variable (`DATABRICKS_TOKEN`).

    Examples:
        `pip install llama-index-embeddings-databricks`

        ```python
        import os
        from llama_index.core import Settings
        from llama_index.embeddings.databricks import DatabricksEmbedding

        # Set up the DatabricksEmbedding class with the required model, API key and serving endpoint
        os.environ["DATABRICKS_TOKEN"] = "<MY TOKEN>"
        os.environ["DATABRICKS_SERVING_ENDPOINT"] = "<MY ENDPOINT>"
        embed_model  = DatabricksEmbedding(model="databricks-bge-large-en")
        Settings.embed_model = embed_model

        # Embed some text
        embeddings = embed_model.get_text_embedding("The DatabricksEmbedding integration works great.")

        ```
    """

    additional_kwargs: Dict[str, Any] = Field(
        default_factory=dict, description="Additional kwargs as for the OpenAI API."
    )

    model: str = Field(
        description="The ID of a model hosted on the databricks endpoint."
    )
    api_key: str = Field(description="The Databricks API key.")
    endpoint: str = Field(description="The Databricks API endpoint.")

    max_retries: int = Field(default=10, description="Maximum number of retries.", ge=0)
    timeout: float = Field(default=60.0, description="Timeout for each request.", ge=0)
    default_headers: Optional[Dict[str, str]] = Field(
        default=None, description="The default headers for API requests."
    )
    reuse_client: bool = Field(
        default=True,
        description=(
            "Reuse the client between requests. When doing anything with large "
            "volumes of async API calls, setting this to false can improve stability."
        ),
    )

    _query_engine: str = PrivateAttr()
    _text_engine: str = PrivateAttr()
    _client: Optional[OpenAI] = PrivateAttr()
    _aclient: Optional[AsyncOpenAI] = PrivateAttr()
    _http_client: Optional[httpx.Client] = PrivateAttr()

    def __init__(
        self,
        model: str,
        endpoint: Optional[str] = None,
        embed_batch_size: int = 100,
        additional_kwargs: Optional[Dict[str, Any]] = None,
        api_key: Optional[str] = None,
        max_retries: int = 10,
        timeout: float = 60.0,
        reuse_client: bool = True,
        callback_manager: Optional[CallbackManager] = None,
        default_headers: Optional[Dict[str, str]] = None,
        http_client: Optional[httpx.Client] = None,
        num_workers: Optional[int] = None,
        **kwargs: Any,
    ) -> None:
        additional_kwargs = additional_kwargs or {}

        api_key = get_from_param_or_env("api_key", api_key, "DATABRICKS_TOKEN")
        endpoint = get_from_param_or_env(
            "endpoint", endpoint, "DATABRICKS_SERVING_ENDPOINT"
        )

        super().__init__(
            model=model,
            endpoint=endpoint,
            embed_batch_size=embed_batch_size,
            callback_manager=callback_manager,
            model_name=model,
            additional_kwargs=additional_kwargs,
            api_key=api_key,
            max_retries=max_retries,
            reuse_client=reuse_client,
            timeout=timeout,
            default_headers=default_headers,
            num_workers=num_workers,
            **kwargs,
        )

        self._client = None
        self._aclient = None
        self._http_client = http_client

    def _get_client(self) -> OpenAI:
        if not self.reuse_client:
            return OpenAI(**self._get_credential_kwargs())

        if self._client is None:
            self._client = OpenAI(**self._get_credential_kwargs())
        return self._client

    def _get_aclient(self) -> AsyncOpenAI:
        if not self.reuse_client:
            return AsyncOpenAI(**self._get_credential_kwargs())

        if self._aclient is None:
            self._aclient = AsyncOpenAI(**self._get_credential_kwargs())
        return self._aclient

    @classmethod
    def class_name(cls) -> str:
        return "DatabricksEmbedding"

    def _get_credential_kwargs(self) -> Dict[str, Any]:
        return {
            "api_key": self.api_key,
            "base_url": self.endpoint,
            "max_retries": self.max_retries,
            "timeout": self.timeout,
            "default_headers": self.default_headers,
            "http_client": self._http_client,
        }

    def _get_query_embedding(self, query: str) -> List[float]:
        """Get query embedding."""
        client = self._get_client()
        return get_embedding(
            client,
            query,
            engine=self.model,
            **self.additional_kwargs,
        )

    async def _aget_query_embedding(self, query: str) -> List[float]:
        """The asynchronous version of _get_query_embedding."""
        aclient = self._get_aclient()
        return await aget_embedding(
            aclient,
            query,
            engine=self.model,
            **self.additional_kwargs,
        )

    def _get_text_embedding(self, text: str) -> List[float]:
        """Get text embedding."""
        client = self._get_client()
        return get_embedding(
            client,
            text,
            engine=self.model,
            **self.additional_kwargs,
        )

    async def _aget_text_embedding(self, text: str) -> List[float]:
        """Asynchronously get text embedding."""
        aclient = self._get_aclient()
        return await aget_embedding(
            aclient,
            text,
            engine=self.model,
            **self.additional_kwargs,
        )

    def _get_text_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Get text embeddings.

        By default, this is a wrapper around _get_text_embedding.
        Can be overridden for batch queries.

        """
        client = self._get_client()
        return get_embeddings(
            client,
            texts,
            engine=self.model,
            **self.additional_kwargs,
        )

    async def _aget_text_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Asynchronously get text embeddings."""
        aclient = self._get_aclient()
        return await aget_embeddings(
            aclient,
            texts,
            engine=self.model,
            **self.additional_kwargs,
        )
