import logging
import warnings
from typing import Dict, Any, Optional, List

import httpx
from llama_index.core.callbacks import CallbackManager
from llama_index.embeddings.openai import OpenAIEmbedding
from pydantic import Field

from llama_index.embeddings.upstage.utils import (
    DEFAULT_UPSTAGE_API_BASE,
    resolve_upstage_credentials,
)

logger = logging.getLogger(__name__)

UPSTAGE_EMBEDDING_MODELS = {
    "solar-1-mini-embedding": {
        "query": "solar-1-mini-embedding-query",
        "passage": "solar-1-mini-embedding-passage",
    },
}


def get_engine(model) -> tuple[Any, Any]:
    """
    get query engine and passage engine for the model.
    """
    if model not in UPSTAGE_EMBEDDING_MODELS:
        raise ValueError(
            f"Unknown model: {model}. Please provide a valid Upstage model name in: {', '.join(UPSTAGE_EMBEDDING_MODELS.keys())}"
        )
    return (
        UPSTAGE_EMBEDDING_MODELS[model]["query"],
        UPSTAGE_EMBEDDING_MODELS[model]["passage"],
    )


class UpstageEmbedding(OpenAIEmbedding):
    """
    Class for Upstage embeddings.

    Args:
        model_name (str): Model for embedding.
        Defaults to "solar-1-mini-embedding".
    """

    additional_kwargs: Dict[str, Any] = Field(
        default_factory=dict, description="Additional kwargs for the Upstage API."
    )

    api_key: str = Field(description="The Upstage API key.")
    api_base: Optional[str] = Field(
        default=DEFAULT_UPSTAGE_API_BASE, description="The base URL for OpenAI API."
    )
    dimensions: Optional[int] = Field(
        None,
        description="Not supported yet. The number of dimensions the resulting output embeddings should have.",
    )

    def __init__(
        self,
        model: str = "solar-1-mini-embedding",
        embed_batch_size: int = 100,
        dimensions: Optional[int] = None,
        additional_kwargs: Dict[str, Any] = None,
        api_key: Optional[str] = None,
        api_base: Optional[str] = None,
        max_retries: int = 10,
        timeout: float = 60.0,
        reuse_client: bool = True,
        callback_manager: Optional[CallbackManager] = None,
        default_headers: Optional[Dict[str, str]] = None,
        http_client: Optional[httpx.Client] = None,
        **kwargs: Any,
    ) -> None:
        additional_kwargs = additional_kwargs or {}
        if dimensions is not None:
            warnings.warn("Received dimensions argument. This is not supported yet.")
            additional_kwargs["dimensions"] = dimensions

        api_key, api_base = resolve_upstage_credentials(
            api_key=api_key, api_base=api_base
        )

        if "model_name" in kwargs:
            model = kwargs.pop("model_name")

        # if model endswith with "-query" or "-passage", remove the suffix and print a warning
        if model.endswith(("-query", "-passage")):
            model = model.rsplit("-", 1)[0]
            logger.warning(
                f"Model name should not end with '-query' or '-passage'. The suffix has been removed. "
                f"Model name: {model}"
            )

        super().__init__(
            embed_batch_size=embed_batch_size,
            dimensions=dimensions,
            callback_manager=callback_manager,
            model_name=model,
            additional_kwargs=additional_kwargs,
            api_key=api_key,
            api_base=api_base,
            max_retries=max_retries,
            reuse_client=reuse_client,
            timeout=timeout,
            default_headers=default_headers,
            **kwargs,
        )

        self._client = None
        self._aclient = None
        self._http_client = http_client

        self._query_engine, self._text_engine = get_engine(model)

    def class_name(cls) -> str:
        return "UpstageEmbedding"

    def _get_credential_kwargs(self) -> Dict[str, Any]:
        return {
            "api_key": self.api_key,
            "base_url": self.api_base,
            "max_retries": self.max_retries,
            "timeout": self.timeout,
            "default_headers": self.default_headers,
            "http_client": self._http_client,
        }

    def _get_query_embedding(self, query: str) -> List[float]:
        """Get query embedding."""
        client = self._get_client()
        text = query.replace("\n", " ")
        return (
            client.embeddings.create(
                input=text, model=self._query_engine, **self.additional_kwargs
            )
            .data[0]
            .embedding
        )

    async def _aget_query_embedding(self, query: str) -> List[float]:
        """The asynchronous version of _get_query_embedding."""
        client = self._get_aclient()
        text = query.replace("\n", " ")
        return (
            (
                await client.embeddings.create(
                    input=text, model=self._query_engine, **self.additional_kwargs
                )
            )
            .data[0]
            .embedding
        )

    def _get_text_embedding(self, text: str) -> List[float]:
        """Get text embedding."""
        client = self._get_client()
        return (
            client.embeddings.create(
                input=text, model=self._text_engine, **self.additional_kwargs
            )
            .data[0]
            .embedding
        )

    async def _aget_text_embedding(self, text: str) -> List[float]:
        """Asynchronously get text embedding."""
        client = self._get_aclient()
        return (
            (
                await client.embeddings.create(
                    input=text, model=self._text_engine, **self.additional_kwargs
                )
            )
            .data[0]
            .embedding
        )

    def _get_text_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Get text embeddings."""
        assert (
            len(texts) <= 2048
        ), "The batch size should be less than or equal to 2048."
        client = self._get_client()
        texts = [text.replace("\n", " ") for text in texts]
        response = []

        for text in texts:
            response.append(
                client.embeddings.create(
                    input=text, model=self._text_engine, **self.additional_kwargs
                )
                .data[0]
                .embedding
            )
        return response

    async def _aget_text_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Asynchronously get text embeddings."""
        assert (
            len(texts) <= 2048
        ), "The batch size should be less than or equal to 2048."
        client = self._get_aclient()
        texts = [text.replace("\n", " ") for text in texts]
        response = []

        for text in texts:
            response.append(
                (
                    await client.embeddings.create(
                        input=text, model=self._text_engine, **self.additional_kwargs
                    )
                )
                .data[0]
                .embedding
            )
        return response
