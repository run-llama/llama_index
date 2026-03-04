import os
from typing import Any, List, Optional
from openai import OpenAI, AsyncOpenAI
from llama_index.core.base.embeddings.base import BaseEmbedding, Embedding
from llama_index.core.bridge.pydantic import Field


class NetmindEmbedding(BaseEmbedding):
    api_base: str = Field(
        default="https://api.netmind.ai/inference-api/openai/v1",
        description="The base URL for the Netmind API.",
    )
    api_key: str = Field(
        default="",
        description="The API key for the Netmind API. If not set, will attempt to use the NETMIND_API_KEY environment variable.",
    )
    timeout: float = Field(
        default=120, description="The timeout for the API request in seconds.", ge=0
    )
    max_retries: int = Field(
        default=3,
        description="The maximum number of retries for the API request.",
        ge=0,
    )

    def __init__(
        self,
        model_name: str,
        timeout: Optional[float] = 120,
        max_retries: Optional[int] = 3,
        api_key: Optional[str] = None,
        api_base: str = "https://api.netmind.ai/inference-api/openai/v1",
        **kwargs: Any,
    ) -> None:
        api_key = api_key or os.environ.get("NETMIND_API_KEY", None)
        super().__init__(
            model_name=model_name,
            api_key=api_key,
            api_base=api_base,
            **kwargs,
        )
        self._client = OpenAI(
            api_key=api_key,
            base_url=self.api_base,
            timeout=timeout,
            max_retries=max_retries,
        )
        self._aclient = AsyncOpenAI(
            api_key=api_key,
            base_url=self.api_base,
            timeout=timeout,
            max_retries=max_retries,
        )

    def _get_text_embedding(self, text: str) -> Embedding:
        """Get text embedding."""
        return (
            self._client.embeddings.create(
                input=[text],
                model=self.model_name,
            )
            .data[0]
            .embedding
        )

    def _get_query_embedding(self, query: str) -> Embedding:
        """Get query embedding."""
        return (
            self._client.embeddings.create(
                input=[query],
                model=self.model_name,
            )
            .data[0]
            .embedding
        )

    def _get_text_embeddings(self, texts: List[str]) -> List[Embedding]:
        """Get text embeddings."""
        data = self._client.embeddings.create(
            input=texts,
            model=self.model_name,
        ).data
        return [d.embedding for d in data]

    async def _aget_text_embedding(self, text: str) -> Embedding:
        """Async get text embedding."""
        return (
            (await self._aclient.embeddings.create(input=[text], model=self.model_name))
            .data[0]
            .embedding
        )

    async def _aget_query_embedding(self, query: str) -> Embedding:
        """Async get query embedding."""
        return (
            (
                await self._aclient.embeddings.create(
                    input=[query], model=self.model_name
                )
            )
            .data[0]
            .embedding
        )

    async def _aget_text_embeddings(self, texts: List[str]) -> List[Embedding]:
        """Async get text embeddings."""
        data = (
            await self._aclient.embeddings.create(
                input=texts,
                model=self.model_name,
            )
        ).data
        return [d.embedding for d in data]
