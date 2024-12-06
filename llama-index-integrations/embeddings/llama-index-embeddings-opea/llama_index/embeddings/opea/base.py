import httpx
from typing import Any, Dict, Optional

from llama_index.core.base.embeddings.base import (
    DEFAULT_EMBED_BATCH_SIZE,
)
from llama_index.core.callbacks import CallbackManager
from llama_index.embeddings.openai import OpenAIEmbedding


class OPEAEmbedding(OpenAIEmbedding):
    """
    OPEA class for embeddings.

    Args:
        model (str): Model for embedding.
        api_base (str): The base URL for OPEA Embeddings microservice.
        additional_kwargs (Dict[str, Any]): Additional kwargs for the OpenAI API.

    Examples:
        `pip install llama-index-embeddings-opea`

        ```python
        from llama_index.embeddings.opea import OPEAEmbedding

        embed_model = OPEAEmbedding(
            model_name="...",
            api_base="http://localhost:8080",
        )
        ```
    """

    def __init__(
        self,
        model_name: str,
        api_base: str,
        dimensions: Optional[int] = None,
        embed_batch_size: int = DEFAULT_EMBED_BATCH_SIZE,
        additional_kwargs: Optional[Dict[str, Any]] = None,
        max_retries: int = 10,
        timeout: float = 60.0,
        reuse_client: bool = True,
        callback_manager: Optional[CallbackManager] = None,
        default_headers: Optional[Dict[str, str]] = None,
        http_client: Optional[httpx.Client] = None,
        api_key: Optional[str] = "fake",
        **kwargs: Any,
    ) -> None:
        super().__init__(
            model_name=model_name,
            dimensions=dimensions,
            embed_batch_size=embed_batch_size,
            additional_kwargs=additional_kwargs,
            api_key=api_key,
            api_base=api_base,
            max_retries=max_retries,
            timeout=timeout,
            reuse_client=reuse_client,
            callback_manager=callback_manager,
            default_headers=default_headers,
            http_client=http_client,
            **kwargs,
        )

    @classmethod
    def class_name(cls) -> str:
        return "OPEAEmbedding"
