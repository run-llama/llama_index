import asyncio
import time
from typing import Any, List

from llama_index.core.base.embeddings.base import BaseEmbedding
from llama_index.core.bridge.pydantic import Field, PrivateAttr


from llama_index.core.bridge.pydantic import Field, PrivateAttr

from llama_index.core.base.llms.generic_utils import get_from_param_or_env

try:
    from alibabacloud_searchplat20240529.models import (
        GetTextEmbeddingRequest,
        GetTextEmbeddingResponse,
    )
    from alibabacloud_tea_openapi.models import Config as AISearchConfig
    from alibabacloud_searchplat20240529.client import Client
    from Tea.exceptions import TeaException
except ImportError:
    raise ImportError(
        "Could not import alibabacloud_searchplat20240529 python package. "
        "Please install it with `pip install alibabacloud-searchplat20240529`."
    )


def retry_decorator(func, wait_seconds: int = 1):
    def wrap(*args, **kwargs):
        while True:
            try:
                return func(*args, **kwargs)
            except TeaException as e:
                if e.code == "Throttling.RateQuota":
                    time.sleep(wait_seconds)
                else:
                    raise

    return wrap


def aretry_decorator(func, wait_seconds: int = 1):
    async def wrap(*args, **kwargs):
        while True:
            try:
                return await func(*args, **kwargs)
            except TeaException as e:
                if e.code == "Throttling.RateQuota":
                    await asyncio.sleep(wait_seconds)
                else:
                    raise

    return wrap


class AlibabaCloudAISearchEmbedding(BaseEmbedding):
    """
    For further details, please visit `https://help.aliyun.com/zh/open-search/search-platform/developer-reference/text-embedding-api-details`.
    """

    _client: Client = PrivateAttr()

    aisearch_api_key: str = Field(default=None, exclude=True)
    endpoint: str = None

    service_id: str = "ops-text-embedding-002"
    workspace_name: str = "default"

    def __init__(
        self, endpoint: str = None, aisearch_api_key: str = None, **kwargs: Any
    ) -> None:
        super().__init__(**kwargs)
        self.aisearch_api_key = get_from_param_or_env(
            "aisearch_api_key", aisearch_api_key, "AISEARCH_API_KEY"
        )
        self.endpoint = get_from_param_or_env("endpoint", endpoint, "AISEARCH_ENDPOINT")

        config = AISearchConfig(
            bearer_token=self.aisearch_api_key,
            endpoint=self.endpoint,
            protocol="http",
        )

        self._client = Client(config=config)

    @classmethod
    def class_name(cls) -> str:
        return "AlibabaCloudAISearchEmbedding"

    @retry_decorator
    def _get_embedding(self, text: str, input_type: str) -> List[float]:
        request = GetTextEmbeddingRequest(input=text, input_type=input_type)
        response: GetTextEmbeddingResponse = self._client.get_text_embedding(
            workspace_name=self.workspace_name,
            service_id=self.service_id,
            request=request,
        )
        embeddings = response.body.result.embeddings
        return embeddings[0].embedding

    @aretry_decorator
    async def _aget_embedding(self, text: str, input_type: str) -> List[float]:
        request = GetTextEmbeddingRequest(input=text, input_type=input_type)
        response: GetTextEmbeddingResponse = (
            await self._client.get_text_embedding_async(
                workspace_name=self.workspace_name,
                service_id=self.service_id,
                request=request,
            )
        )
        embeddings = response.body.result.embeddings
        return embeddings[0].embedding

    @retry_decorator
    def _get_embeddings(self, texts: List[str], input_type: str) -> List[List[float]]:
        request = GetTextEmbeddingRequest(input=texts, input_type=input_type)
        response: GetTextEmbeddingResponse = self._client.get_text_embedding(
            workspace_name=self.workspace_name,
            service_id=self.service_id,
            request=request,
        )
        embeddings = response.body.result.embeddings
        return [emb.embedding for emb in embeddings]

    @aretry_decorator
    async def _aget_embeddings(
        self,
        texts: List[str],
        input_type: str,
    ) -> List[List[float]]:
        request = GetTextEmbeddingRequest(input=texts, input_type=input_type)
        response: GetTextEmbeddingResponse = (
            await self._client.get_text_embedding_async(
                workspace_name=self.workspace_name,
                service_id=self.service_id,
                request=request,
            )
        )
        embeddings = response.body.result.embeddings
        return [emb.embedding for emb in embeddings]

    def _get_query_embedding(self, query: str) -> List[float]:
        """Get query embedding."""
        return self._get_embedding(
            query,
            input_type="query",
        )

    async def _aget_query_embedding(self, query: str) -> List[float]:
        """The asynchronous version of _get_query_embedding."""
        return await self._aget_embedding(
            query,
            input_type="query",
        )

    def _get_text_embedding(self, text: str) -> List[float]:
        """Get text embedding."""
        return self._get_embedding(
            text,
            input_type="document",
        )

    async def _aget_text_embedding(self, text: str) -> List[float]:
        """The asynchronous version of _get_text_embedding."""
        return await self._aget_embedding(
            text,
            input_type="document",
        )

    def _get_text_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Get text embeddings."""
        return self._get_embeddings(
            texts,
            input_type="document",
        )

    async def _aget_text_embeddings(self, texts: List[str]) -> List[List[float]]:
        """The asynchronous version of _get_text_embeddings."""
        return await self._aget_embeddings(
            texts,
            input_type="document",
        )
