import logging
import os
from typing import Any, List, Optional

import httpx
from llama_index.core.base.embeddings.base import BaseEmbedding
from llama_index.core.bridge.pydantic import Field, PrivateAttr
from llama_index.core.callbacks.base import CallbackManager
from mixedbread_ai import TruncationStrategy
from mixedbread_ai.client import MixedbreadAI, AsyncMixedbreadAI
from mixedbread_ai.core import RequestOptions
from mixedbread_ai.types import EncodingFormat

logger = logging.getLogger(__name__)


class MixedbreadAIEmbedding(BaseEmbedding):
    """
    Class to get embeddings using the mixedbread ai embedding API with models such as 'mixedbread-ai/mxbai-embed-large-v1'.

    Args:
        api_key (Optional[str]): mixedbread ai API key. Defaults to None.
        model_name (str): Model for embedding. Defaults to "mixedbread-ai/mxbai-embed-large-v1".
        encoding_format (EncodingFormat): Encoding format for embeddings. Defaults to EncodingFormat.FLOAT.
        truncation_strategy (TruncationStrategy): Truncation strategy. Defaults to TruncationStrategy.START.
        normalized (bool): Whether to normalize the embeddings. Defaults to True.
        dimensions (Optional[int]): Number of dimensions for embeddings. Only applicable for models with matryoshka support.
        prompt (Optional[str]): An optional prompt to provide context to the model.
        embed_batch_size (Optional[int]): The batch size for embedding calls. Defaults to 128.
        callback_manager (Optional[CallbackManager]): Manager for handling callbacks.
        timeout (Optional[float]): Timeout for API calls.
        max_retries (Optional[int]): Maximum number of retries for API calls.
        httpx_client (Optional[httpx.Client]): Custom HTTPX client.
        httpx_async_client (Optional[httpx.AsyncClient]): Custom asynchronous HTTPX client.

    """

    api_key: str = Field(description="The mixedbread ai API key.", min_length=1)
    model_name: str = Field(
        default="mixedbread-ai/mxbai-embed-large-v1",
        description="Model to use for embeddings.",
        min_length=1,
    )
    encoding_format: EncodingFormat = Field(
        default=EncodingFormat.FLOAT, description="Encoding format for the embeddings."
    )
    truncation_strategy: TruncationStrategy = Field(
        default=TruncationStrategy.START,
        description="Truncation strategy for input text.",
    )
    normalized: bool = Field(
        default=True, description="Whether to normalize the embeddings."
    )
    dimensions: Optional[int] = Field(
        default=None,
        description="Number of dimensions for embeddings. Only applicable for models with matryoshka support.",
        gt=0,
    )
    prompt: Optional[str] = Field(
        default=None,
        description="An optional prompt to provide context to the model.",
        min_length=1,
    )
    embed_batch_size: int = Field(
        default=128, description="The batch size for embedding calls.", gt=0, le=256
    )

    _client: MixedbreadAI = PrivateAttr()
    _async_client: AsyncMixedbreadAI = PrivateAttr()
    _request_options: Optional[RequestOptions] = PrivateAttr()

    def __init__(
        self,
        api_key: Optional[str] = None,
        model_name: str = "mixedbread-ai/mxbai-embed-large-v1",
        encoding_format: EncodingFormat = EncodingFormat.FLOAT,
        truncation_strategy: TruncationStrategy = TruncationStrategy.START,
        normalized: bool = True,
        dimensions: Optional[int] = None,
        prompt: Optional[str] = None,
        embed_batch_size: Optional[int] = None,
        callback_manager: Optional[CallbackManager] = None,
        timeout: Optional[float] = None,
        max_retries: Optional[int] = None,
        httpx_client: Optional[httpx.Client] = None,
        httpx_async_client: Optional[httpx.AsyncClient] = None,
        **kwargs: Any,
    ):
        if embed_batch_size is None:
            embed_batch_size = 128  # Default batch size for mixedbread ai

        try:
            api_key = api_key or os.environ["MXBAI_API_KEY"]
        except KeyError:
            raise ValueError(
                "Must pass in mixedbread ai API key or "
                "specify via MXBAI_API_KEY environment variable "
            )

        super().__init__(
            api_key=api_key,
            model_name=model_name,
            encoding_format=encoding_format,
            truncation_strategy=truncation_strategy,
            normalized=normalized,
            dimensions=dimensions,
            prompt=prompt,
            embed_batch_size=embed_batch_size,
            callback_manager=callback_manager,
            **kwargs,
        )

        self._client = MixedbreadAI(
            api_key=api_key, timeout=timeout, httpx_client=httpx_client
        )
        self._async_client = AsyncMixedbreadAI(
            api_key=api_key, timeout=timeout, httpx_client=httpx_async_client
        )

        self._request_options = (
            RequestOptions(max_retries=max_retries) if max_retries is not None else None
        )

    @classmethod
    def class_name(cls) -> str:
        return "MixedbreadAIEmbedding"

    def _get_embedding(self, texts: List[str]) -> List[List[float]]:
        """
        Get embeddings for a list of texts using the mixedbread ai API.

        Args:
            texts (List[str]): List of texts to embed.

        Returns:
            List[List[float]]: List of embeddings.

        """
        response = self._client.embeddings(
            model=self.model_name,
            input=texts,
            encoding_format=self.encoding_format,
            normalized=self.normalized,
            truncation_strategy=self.truncation_strategy,
            dimensions=self.dimensions,
            prompt=self.prompt,
            request_options=self._request_options,
        )
        return [item.embedding for item in response.data]

    async def _aget_embedding(self, texts: List[str]) -> List[List[float]]:
        """
        Asynchronously get embeddings for a list of texts using the mixedbread ai API.

        Args:
            texts (List[str]): List of texts to embed.

        Returns:
            List[List[float]]: List of embeddings.

        """
        response = await self._async_client.embeddings(
            model=self.model_name,
            input=texts,
            encoding_format=self.encoding_format,
            normalized=self.normalized,
            truncation_strategy=self.truncation_strategy,
            dimensions=self.dimensions,
            prompt=self.prompt,
            request_options=self._request_options,
        )
        return [item.embedding for item in response.data]

    def _get_query_embedding(self, query: str) -> List[float]:
        """
        Get embedding for a query using the mixedbread ai API.

        Args:
            query (str): Query text.

        Returns:
            List[float]: Embedding for the query.

        """
        return self._get_embedding([query])[0]

    async def _aget_query_embedding(self, query: str) -> List[float]:
        """
        Asynchronously get embedding for a query using the mixedbread ai API.

        Args:
            query (str): Query text.

        Returns:
            List[float]: Embedding for the query.

        """
        r = await self._aget_embedding([query])
        return r[0]

    def _get_text_embedding(self, text: str) -> List[float]:
        """
        Get embedding for a text using the mixedbread ai API.

        Args:
            text (str): Text to embed.

        Returns:
            List[float]: Embedding for the text.

        """
        return self._get_embedding([text])[0]

    async def _aget_text_embedding(self, text: str) -> List[float]:
        """
        Asynchronously get embedding for a text using the mixedbread ai API.

        Args:
            text (str): Text to embed.

        Returns:
            List[float]: Embedding for the text.

        """
        r = await self._aget_embedding([text])
        return r[0]

    def _get_text_embeddings(self, texts: List[str]) -> List[List[float]]:
        """
        Get embeddings for multiple texts using the mixedbread ai API.

        Args:
            texts (List[str]): List of texts to embed.

        Returns:
            List[List[float]]: List of embeddings.

        """
        return self._get_embedding(texts)

    async def _aget_text_embeddings(self, texts: List[str]) -> List[List[float]]:
        """
        Asynchronously get embeddings for multiple texts using the mixedbread ai API.

        Args:
            texts (List[str]): List of texts to embed.

        Returns:
            List[List[float]]: List of embeddings.

        """
        return await self._aget_embedding(texts)
