"""Telnyx Embeddings integration for LlamaIndex.

Telnyx provides an OpenAI-compatible Embeddings API.
This integration wraps the OpenAILikeEmbedding base class with Telnyx-specific defaults.
"""

import os
from typing import Any, Dict, Optional

import httpx
from llama_index.core.callbacks.base import CallbackManager
from llama_index.embeddings.openai_like import OpenAILikeEmbedding

DEFAULT_TELNYX_API_BASE = "https://api.telnyx.com/v2/ai/openai"
DEFAULT_TELNYX_EMBEDDING_MODEL = "thenlper/gte-large"


class TelnyxEmbedding(OpenAILikeEmbedding):
    """Telnyx Embeddings.

    Telnyx provides an OpenAI-compatible Embeddings API with models
    like gte-large and e5-large.

    To use, set the ``TELNYX_API_KEY`` environment variable or pass it
    directly via the ``api_key`` parameter.

    Examples:
        `pip install llama-index-embeddings-telnyx`

        ```python
        from llama_index.embeddings.telnyx import TelnyxEmbedding

        embed_model = TelnyxEmbedding(
            model_name="thenlper/gte-large",
            api_key="your_api_key",
        )

        embedding = embed_model.get_text_embedding("What is Telnyx?")
        print(len(embedding))
        ```
    """

    def __init__(
        self,
        model_name: str = DEFAULT_TELNYX_EMBEDDING_MODEL,
        api_key: Optional[str] = None,
        api_base: Optional[str] = None,
        embed_batch_size: int = 10,
        dimensions: Optional[int] = None,
        additional_kwargs: Optional[Dict[str, Any]] = None,
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
        api_key = api_key or os.environ.get("TELNYX_API_KEY")
        if not api_key:
            raise ValueError(
                "Telnyx API key is required. Set the TELNYX_API_KEY environment "
                "variable or pass api_key directly."
            )

        api_base = (
            api_base
            or os.environ.get("TELNYX_API_BASE")
            or DEFAULT_TELNYX_API_BASE
        )

        # Add Telnyx user-agent for telemetry
        headers = dict(default_headers or {})
        headers.setdefault(
            "User-Agent", "llama-index-embeddings-telnyx/0.1.0"
        )

        super().__init__(
            model_name=model_name,
            api_key=api_key,
            api_base=api_base,
            embed_batch_size=embed_batch_size,
            dimensions=dimensions,
            additional_kwargs=additional_kwargs,
            max_retries=max_retries,
            timeout=timeout,
            reuse_client=reuse_client,
            callback_manager=callback_manager,
            default_headers=headers,
            http_client=http_client,
            async_http_client=async_http_client,
            num_workers=num_workers,
            **kwargs,
        )

    @classmethod
    def class_name(cls) -> str:
        """Get class name."""
        return "TelnyxEmbedding"
