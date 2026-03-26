from typing import Any, Dict, Optional

import httpx
from llama_index.core.constants import DEFAULT_EMBED_BATCH_SIZE
from llama_index.core.bridge.pydantic import Field
from llama_index.core.callbacks import CallbackManager
from llama_index.core.base.llms.generic_utils import get_from_param_or_env
from llama_index.embeddings.openai import OpenAIEmbedding

DEFAULT_API_BASE = (
    "https://model-{model_id}.api.baseten.co/environments/production/sync/v1"
)


class BasetenEmbedding(OpenAIEmbedding):
    """
    Baseten class for embeddings.

    Args:
        model_id (str): The Baseten model ID (e.g., "03y7n6e3").
        api_key (Optional[str]): The Baseten API key.
        embed_batch_size (int): The batch size for embedding calls.
        additional_kwargs (Optional[Dict[str, Any]]): Additional kwargs for the API.
        max_retries (int): The maximum number of retries to make.
        timeout (float): Timeout for each request.
        callback_manager (Optional[CallbackManager]): Callback manager for logging.
        default_headers (Optional[Dict[str, str]]): Default headers for API requests.

    Examples:
        ```python
        from llama_index.embeddings.baseten import BasetenEmbedding

        # Using dedicated endpoint
        # You can find the model_id by in the Baseten dashboard here: https://app.baseten.co/overview
        embed_model = BasetenEmbedding(
            model_id="MODEL_ID,
            api_key="YOUR_API_KEY",
        )

        # Single embedding
        embedding = embed_model.get_text_embedding("Hello, world!")

        # Batch embeddings
        embeddings = embed_model.get_text_embedding_batch([
            "Hello, world!",
            "Goodbye, world!"
        ])
        ```

    """

    additional_kwargs: Dict[str, Any] = Field(
        default_factory=dict, description="Additional kwargs for the OpenAI API."
    )

    api_key: str = Field(description="The Baseten API key.")
    api_base: str = Field(default="", description="The base URL for Baseten API.")
    api_version: str = Field(default="", description="The version for OpenAI API.")

    def __init__(
        self,
        model_id: str,
        dimensions: Optional[int] = None,
        embed_batch_size: int = DEFAULT_EMBED_BATCH_SIZE,
        additional_kwargs: Optional[Dict[str, Any]] = None,
        api_key: Optional[str] = None,
        api_base: Optional[str] = None,
        api_version: Optional[str] = None,
        max_retries: int = 10,
        timeout: float = 60.0,
        reuse_client: bool = True,
        callback_manager: Optional[CallbackManager] = None,
        default_headers: Optional[Dict[str, str]] = None,
        http_client: Optional[httpx.Client] = None,
        **kwargs: Any,
    ) -> None:
        # Use the dedicated endpoint URL format
        api_base = DEFAULT_API_BASE.format(model_id=model_id)
        api_key = get_from_param_or_env("api_key", api_key, "BASETEN_API_KEY")

        super().__init__(
            model_name=model_id,
            dimensions=dimensions,
            embed_batch_size=embed_batch_size,
            additional_kwargs=additional_kwargs,
            api_key=api_key,
            api_base=api_base,
            api_version=api_version,
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
        return "BasetenEmbedding"
