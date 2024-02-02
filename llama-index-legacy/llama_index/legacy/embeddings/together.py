import asyncio
import os
from typing import Any, List, Optional

import httpx
import requests

from llama_index.bridge.pydantic import Field
from llama_index.embeddings.base import BaseEmbedding, Embedding


class TogetherEmbedding(BaseEmbedding):
    api_base: str = Field(
        default="https://api.together.xyz/v1",
        description="The base URL for the Together API.",
    )
    api_key: str = Field(
        default="",
        description="The API key for the Together API. If not set, will attempt to use the TOGETHER_API_KEY environment variable.",
    )

    def __init__(
        self,
        model_name: str,
        api_key: Optional[str] = None,
        api_base: str = "https://api.together.xyz/v1",
        **kwargs: Any,
    ) -> None:
        api_key = api_key or os.environ.get("TOGETHER_API_KEY", None)
        super().__init__(
            model_name=model_name,
            api_key=api_key,
            api_base=api_base,
            **kwargs,
        )

    def _generate_embedding(self, text: str, model_api_string: str) -> Embedding:
        """Generate embeddings from Together API.

        Args:
            text: str. An input text sentence or document.
            model_api_string: str. An API string for a specific embedding model of your choice.

        Returns:
            embeddings: a list of float numbers. Embeddings correspond to your given text.
        """
        headers = {
            "accept": "application/json",
            "content-type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }

        session = requests.Session()
        response = session.post(
            self.api_base.strip("/") + "/embeddings",
            headers=headers,
            json={"input": text, "model": model_api_string},
        )
        if response.status_code != 200:
            raise ValueError(
                f"Request failed with status code {response.status_code}: {response.text}"
            )

        return response.json()["data"][0]["embedding"]

    async def _agenerate_embedding(self, text: str, model_api_string: str) -> Embedding:
        """Async generate embeddings from Together API.

        Args:
            text: str. An input text sentence or document.
            model_api_string: str. An API string for a specific embedding model of your choice.

        Returns:
            embeddings: a list of float numbers. Embeddings correspond to your given text.
        """
        headers = {
            "accept": "application/json",
            "content-type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }

        async with httpx.AsyncClient() as client:
            response = await client.post(
                self.api_base.strip("/") + "/embeddings",
                headers=headers,
                json={"input": text, "model": model_api_string},
            )
            if response.status_code != 200:
                raise ValueError(
                    f"Request failed with status code {response.status_code}: {response.text}"
                )

            return response.json()["data"][0]["embedding"]

    def _get_text_embedding(self, text: str) -> Embedding:
        """Get text embedding."""
        return self._generate_embedding(text, self.model_name)

    def _get_query_embedding(self, query: str) -> Embedding:
        """Get query embedding."""
        return self._generate_embedding(query, self.model_name)

    def _get_text_embeddings(self, texts: List[str]) -> List[Embedding]:
        """Get text embeddings."""
        return [self._generate_embedding(text, self.model_name) for text in texts]

    async def _aget_text_embedding(self, text: str) -> Embedding:
        """Async get text embedding."""
        return await self._agenerate_embedding(text, self.model_name)

    async def _aget_query_embedding(self, query: str) -> Embedding:
        """Async get query embedding."""
        return await self._agenerate_embedding(query, self.model_name)

    async def _aget_text_embeddings(self, texts: List[str]) -> List[Embedding]:
        """Async get text embeddings."""
        return await asyncio.gather(
            *[self._agenerate_embedding(text, self.model_name) for text in texts]
        )
