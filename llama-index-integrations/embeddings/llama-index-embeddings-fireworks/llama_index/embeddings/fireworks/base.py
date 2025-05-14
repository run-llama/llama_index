from typing import Any, Dict, Optional

import httpx
from llama_index.core.base.embeddings.base import (
    DEFAULT_EMBED_BATCH_SIZE,
)
from llama_index.core.bridge.pydantic import Field
from llama_index.core.callbacks import CallbackManager
from llama_index.embeddings.fireworks.utils import (
    resolve_fireworks_credentials,
)
from llama_index.embeddings.openai import OpenAIEmbedding

DEFAULT_API_BASE = "https://api.fireworks.ai/inference/v1"
DEFAULT_MODEL = "nomic-ai/nomic-embed-text-v1.5"


class FireworksEmbedding(OpenAIEmbedding):
    """
    Fireworks class for embeddings.

    Args:
        model (str): Model for embedding.
            Defaults to "nomic-ai/nomic-embed-text-v1.5"

    """

    additional_kwargs: Dict[str, Any] = Field(
        default_factory=dict, description="Additional kwargs for the OpenAI API."
    )

    api_key: str = Field(description="The Fireworks API key.")
    api_base: str = Field(description="The base URL for Fireworks API.")
    api_version: str = Field(description="The version for OpenAI API.")

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL,
        dimensions: Optional[int] = None,
        embed_batch_size: int = DEFAULT_EMBED_BATCH_SIZE,
        additional_kwargs: Optional[Dict[str, Any]] = None,
        api_key: Optional[str] = None,
        api_base: Optional[str] = DEFAULT_API_BASE,
        api_version: Optional[str] = None,
        max_retries: int = 10,
        timeout: float = 60.0,
        reuse_client: bool = True,
        callback_manager: Optional[CallbackManager] = None,
        default_headers: Optional[Dict[str, str]] = None,
        http_client: Optional[httpx.Client] = None,
        **kwargs: Any,
    ) -> None:
        api_key, api_base, api_version = resolve_fireworks_credentials(
            api_key=api_key,
            api_base=api_base,
            api_version=api_version,
        )

        super().__init__(
            model_name=model_name,
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
        return "FireworksEmbedding"
