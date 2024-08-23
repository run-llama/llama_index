import aiohttp
import asyncio
import requests
from typing import Any, List
from llama_index.core.base.embeddings.base import BaseEmbedding
from llama_index.core.bridge.pydantic import Field


class XinferenceEmbedding(BaseEmbedding):
    """Class for Xinference embeddings."""

    model_uid: str = Field(
        default="unknown",
        description="The Xinference model uid to use.",
    )
    base_url: str = Field(
        default="http://localhost:9997",
        description="The Xinference base url to use.",
    )
    timeout: float = Field(
        default=60.0,
        description="Timeout in seconds for the request.",
    )

    def __init__(
        self,
        model_uid: str,
        base_url: str = "http://localhost:9997",
        timeout: float = 60.0,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            model_uid=model_uid,
            base_url=base_url,
            timeout=timeout,
            **kwargs,
        )

    @classmethod
    def class_name(cls) -> str:
        return "XinferenceEmbedding"

    def _get_query_embedding(self, query: str) -> List[float]:
        """Get query embedding."""
        return self.get_general_text_embedding(query)

    async def _aget_query_embedding(self, query: str) -> List[float]:
        """The asynchronous version of _get_query_embedding."""
        return await self.aget_general_text_embedding(query)

    def _get_text_embedding(self, text: str) -> List[float]:
        """Get text embedding."""
        return self.get_general_text_embedding(text)

    async def _aget_text_embedding(self, text: str) -> List[float]:
        """Asynchronously get text embedding."""
        return await self.aget_general_text_embedding(text)

    def _get_text_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Get text embeddings."""
        embeddings_list: List[List[float]] = []
        for text in texts:
            embeddings = self.get_general_text_embedding(text)
            embeddings_list.append(embeddings)
        return embeddings_list

    async def _aget_text_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Asynchronously get text embeddings."""
        return await asyncio.gather(
            *[self.aget_general_text_embedding(text) for text in texts]
        )

    def get_general_text_embedding(self, prompt: str) -> List[float]:
        """Get Xinference embeddings."""
        headers = {"Content-Type": "application/json"}
        json_data = {"input": prompt, "model": self.model_uid}
        response = requests.post(
            url=f"{self.base_url}/v1/embeddings",
            headers=headers,
            json=json_data,
            timeout=self.timeout,
        )
        response.encoding = "utf-8"
        if response.status_code != 200:
            raise Exception(
                f"Xinference call failed with status code {response.status_code}."
                f"Details: {response.text}"
            )
        return response.json()["data"][0]["embedding"]

    async def aget_general_text_embedding(self, prompt: str) -> List[float]:
        """Asynchronously get Xinference embeddings."""
        headers = {"Content-Type": "application/json"}
        json_data = {"input": prompt, "model": self.model_uid}
        async with aiohttp.ClientSession() as session:
            async with session.post(
                url=f"{self.base_url}/v1/embeddings",
                headers=headers,
                json=json_data,
                timeout=self.timeout,
            ) as response:
                if response.status != 200:
                    raise Exception(
                        f"Xinference call failed with status code {response.status}."
                    )
                data = await response.json()
                return data["data"][0]["embedding"]
