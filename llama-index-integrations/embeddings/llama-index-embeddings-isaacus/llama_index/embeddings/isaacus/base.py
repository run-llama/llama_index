"""Isaacus embeddings file."""

import logging
from typing import Any, List, Literal, Optional

import isaacus

from llama_index.core.base.embeddings.base import (
    DEFAULT_EMBED_BATCH_SIZE,
    BaseEmbedding,
    Embedding,
)
from llama_index.core.bridge.pydantic import Field, PrivateAttr
from llama_index.core.callbacks.base import CallbackManager
from llama_index.core.base.llms.generic_utils import get_from_param_or_env

logger = logging.getLogger(__name__)

DEFAULT_ISAACUS_API_BASE = "https://api.isaacus.com/v1"
DEFAULT_ISAACUS_MODEL = "kanon-2-embedder"


class IsaacusEmbedding(BaseEmbedding):
    """
    Isaacus Embeddings Integration.

    This class provides an interface to Isaacus' embedding API, featuring the
    Kanon 2 Embedder - the world's most accurate legal embedding model on the
    Massive Legal Embedding Benchmark (MLEB).

    Args:
        model (str, optional): The model to use. Defaults to "kanon-2-embedder".
        api_key (str, optional): The API key for Isaacus. Defaults to ISAACUS_API_KEY.
        base_url (str, optional): The base URL for Isaacus API. Defaults to ISAACUS_BASE_URL.
        dimensions (int, optional): The desired embedding dimensionality.
        task (str, optional): Task type: "retrieval/query" or "retrieval/document".
        overflow_strategy (str, optional): Strategy for handling overflow. Defaults to "drop_end".
        timeout (float, optional): Timeout for requests in seconds. Defaults to 60.0.
        **kwargs: Additional keyword arguments.

    Environment Variables:
        - ISAACUS_API_KEY: The API key for Isaacus
        - ISAACUS_BASE_URL: The base URL for Isaacus API (optional)

    Raises:
        ValueError: If required environment variables are not set.

    """

    model: str = Field(
        default=DEFAULT_ISAACUS_MODEL,
        description="The model to use for embeddings.",
    )
    api_key: Optional[str] = Field(default=None, description="The API key for Isaacus.")
    base_url: Optional[str] = Field(
        default=None, description="The base URL for Isaacus API."
    )
    dimensions: Optional[int] = Field(
        default=None, description="The desired embedding dimensionality."
    )
    task: Optional[Literal["retrieval/query", "retrieval/document"]] = Field(
        default=None,
        description="Task type: 'retrieval/query' or 'retrieval/document'.",
    )
    overflow_strategy: Optional[Literal["drop_end"]] = Field(
        default="drop_end", description="Strategy for handling overflow."
    )
    timeout: float = Field(default=60.0, description="Timeout for requests in seconds.")

    _client: Any = PrivateAttr()
    _aclient: Any = PrivateAttr()

    def __init__(
        self,
        model: str = DEFAULT_ISAACUS_MODEL,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        dimensions: Optional[int] = None,
        task: Optional[Literal["retrieval/query", "retrieval/document"]] = None,
        overflow_strategy: Optional[Literal["drop_end"]] = "drop_end",
        timeout: float = 60.0,
        embed_batch_size: int = DEFAULT_EMBED_BATCH_SIZE,
        callback_manager: Optional[CallbackManager] = None,
        **kwargs: Any,
    ) -> None:
        """
        Initialize an instance of the IsaacusEmbedding class.

        Args:
            model (str, optional): The model to use. Defaults to "kanon-2-embedder".
            api_key (str, optional): The API key for Isaacus. Defaults to ISAACUS_API_KEY.
            base_url (str, optional): The base URL for Isaacus API.
            dimensions (int, optional): The desired embedding dimensionality.
            task (str, optional): Task type: "retrieval/query" or "retrieval/document".
            overflow_strategy (str, optional): Strategy for handling overflow.
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
                "ISAACUS_API_KEY",
            )
        except ValueError:
            raise ValueError(
                "API key is required. Set ISAACUS_API_KEY environment variable or pass api_key parameter."
            )

        # Get base URL from parameter or environment (optional)
        if base_url is None:
            try:
                base_url = get_from_param_or_env(
                    "base_url",
                    base_url,
                    "ISAACUS_BASE_URL",
                )
            except ValueError:
                base_url = DEFAULT_ISAACUS_API_BASE

        super().__init__(
            model_name=model,
            model=model,
            api_key=api_key,
            base_url=base_url,
            dimensions=dimensions,
            task=task,
            overflow_strategy=overflow_strategy,
            timeout=timeout,
            embed_batch_size=embed_batch_size,
            callback_manager=callback_manager,
            **kwargs,
        )

        # Initialize Isaacus clients
        self._client = isaacus.Isaacus(
            api_key=self.api_key,
            base_url=self.base_url,
            timeout=self.timeout,
        )
        self._aclient = isaacus.AsyncIsaacus(
            api_key=self.api_key,
            base_url=self.base_url,
            timeout=self.timeout,
        )

    @classmethod
    def class_name(cls) -> str:
        """Get class name."""
        return "IsaacusEmbedding"

    def _prepare_request_params(
        self, text: str, task_override: Optional[str] = None
    ) -> dict:
        """Prepare request parameters for the Isaacus API."""
        params = {
            "model": self.model,
            "texts": text,
        }

        # Use task_override if provided, otherwise use instance task
        task_to_use = task_override if task_override is not None else self.task
        if task_to_use is not None:
            params["task"] = task_to_use

        if self.dimensions is not None:
            params["dimensions"] = self.dimensions

        if self.overflow_strategy is not None:
            params["overflow_strategy"] = self.overflow_strategy

        return params

    def _get_query_embedding(self, query: str) -> Embedding:
        """
        Get query embedding.

        For queries, we use the 'retrieval/query' task if no task is explicitly set.
        """
        return self._get_text_embedding(query, task_override="retrieval/query")

    def _get_text_embedding(
        self, text: str, task_override: Optional[str] = None
    ) -> Embedding:
        """Get text embedding."""
        try:
            params = self._prepare_request_params(text, task_override)
            response = self._client.embeddings.create(**params)

            # Extract the embedding from the response
            if response.embeddings and len(response.embeddings) > 0:
                return response.embeddings[0].embedding
            else:
                raise ValueError("No embeddings returned from API")

        except Exception as e:
            logger.error(f"Error while embedding text: {e}")
            raise ValueError(f"Unable to embed text: {e}")

    def _get_text_embeddings(self, texts: List[str]) -> List[Embedding]:
        """
        Get embeddings for multiple texts.

        Note: The Isaacus API supports batch embedding, so we send all texts at once.
        """
        try:
            params = self._prepare_request_params(texts, task_override=self.task)
            response = self._client.embeddings.create(**params)

            # Extract embeddings from response, maintaining order
            embeddings = []
            for emb_obj in sorted(response.embeddings, key=lambda x: x.index):
                embeddings.append(emb_obj.embedding)

            return embeddings

        except Exception as e:
            logger.error(f"Error while embedding texts: {e}")
            raise ValueError(f"Unable to embed texts: {e}")

    async def _aget_query_embedding(self, query: str) -> Embedding:
        """
        Get query embedding asynchronously.

        For queries, we use the 'retrieval/query' task if no task is explicitly set.
        """
        return await self._aget_text_embedding(query, task_override="retrieval/query")

    async def _aget_text_embedding(
        self, text: str, task_override: Optional[str] = None
    ) -> Embedding:
        """Get text embedding asynchronously."""
        try:
            params = self._prepare_request_params(text, task_override)
            response = await self._aclient.embeddings.create(**params)

            # Extract the embedding from the response
            if response.embeddings and len(response.embeddings) > 0:
                return response.embeddings[0].embedding
            else:
                raise ValueError("No embeddings returned from API")

        except Exception as e:
            logger.error(f"Error while embedding text: {e}")
            raise ValueError(f"Unable to embed text: {e}")

    async def _aget_text_embeddings(self, texts: List[str]) -> List[Embedding]:
        """
        Get embeddings for multiple texts asynchronously.

        Note: The Isaacus API supports batch embedding, so we send all texts at once.
        """
        try:
            params = self._prepare_request_params(texts, task_override=self.task)
            response = await self._aclient.embeddings.create(**params)

            # Extract embeddings from response, maintaining order
            embeddings = []
            for emb_obj in sorted(response.embeddings, key=lambda x: x.index):
                embeddings.append(emb_obj.embedding)

            return embeddings

        except Exception as e:
            logger.error(f"Error while embedding texts: {e}")
            raise ValueError(f"Unable to embed texts: {e}")
