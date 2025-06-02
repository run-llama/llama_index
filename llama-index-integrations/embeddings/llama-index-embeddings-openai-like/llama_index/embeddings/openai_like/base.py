"""OpenAI-Like embeddings."""

from typing import Any, Dict, Optional

import httpx
from llama_index.core.callbacks.base import CallbackManager
from llama_index.embeddings.openai import OpenAIEmbedding


class OpenAILikeEmbedding(OpenAIEmbedding):
    """
    OpenAI-Like class for embeddings.

    Args:
        model_name (str):
            Model for embedding.
        api_key (str):
            The API key (if any) to use for the embedding API.
        api_base (str):
            The base URL for the embedding API.
        api_version (str):
            The version for the embedding API.
        max_retries (int):
            The maximum number of retries for the embedding API.
        timeout (float):
            The timeout for the embedding API.
        reuse_client (bool):
            Whether to reuse the client for the embedding API.
        callback_manager (CallbackManager):
            The callback manager for the embedding API.
        default_headers (Dict[str, str]):
            The default headers for the embedding API.
        additional_kwargs (Dict[str, Any]):
            Additional kwargs for the embedding API.
        dimensions (int):
            The number of dimensions for the embedding API.

    Example:
        ```bash
        pip install llama-index-embeddings-openai-like
        ```

        ```python
        from llama_index.embeddings.openai_like import OpenAILikeEmbedding

        embedding = OpenAILikeEmbedding(
            model_name="my-model-name",
            api_base="http://localhost:1234/v1",
            api_key="fake",
            embed_batch_size=10,
        )
        ```

    """

    def __init__(
        self,
        model_name: str,
        embed_batch_size: int = 10,
        dimensions: Optional[int] = None,
        additional_kwargs: Optional[Dict[str, Any]] = None,
        api_key: str = "fake",
        api_base: Optional[str] = None,
        api_version: Optional[str] = None,
        max_retries: int = 10,
        timeout: float = 60.0,
        reuse_client: bool = True,
        callback_manager: Optional[CallbackManager] = None,
        default_headers: Optional[Dict[str, str]] = None,
        http_client: Optional[httpx.Client] = None,
        async_http_client: Optional[httpx.AsyncClient] = None,
        num_workers: Optional[int] = None,
        **kwargs: Any,
    ) -> None:
        # ensure model is not passed in kwargs, will cause error in parent class
        if "model" in kwargs:
            raise ValueError(
                "Use `model_name` instead of `model` to initialize OpenAILikeEmbedding"
            )

        super().__init__(
            model_name=model_name,
            embed_batch_size=embed_batch_size,
            dimensions=dimensions,
            callback_manager=callback_manager,
            additional_kwargs=additional_kwargs,
            api_key=api_key,
            api_base=api_base,
            api_version=api_version,
            max_retries=max_retries,
            reuse_client=reuse_client,
            timeout=timeout,
            default_headers=default_headers,
            http_client=http_client,
            async_http_client=async_http_client,
            num_workers=num_workers,
            **kwargs,
        )
