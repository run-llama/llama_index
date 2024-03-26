"""NeMo embeddings file."""

import json
import requests
import aiohttp
from typing import Any, List, Optional

from llama_index.core.base.embeddings.base import (
    DEFAULT_EMBED_BATCH_SIZE,
    BaseEmbedding,
)
from llama_index.core.bridge.pydantic import Field, PrivateAttr
from llama_index.core.callbacks.base import CallbackManager


class NeMoEmbedding(BaseEmbedding):
    """Nvidia NeMo embeddings."""

    model_name: str = Field(
        default="NV-Embed-QA-003",
        description="Name of the NeMo embeddings model to use.\n"
        "Defaults to 'NV-Embed-QA-003'.\n",
    )
    api_endpoint_url: str = Field(
        default="http://localhost:8088/v1/embeddings",
        description="Endpoint of NeMo Embeddings microservice to use",
    )

    _api_endpoint_url: str = PrivateAttr()

    def __init__(
        self,
        model_name: str = "NV-Embed-QA-003",
        api_endpoint_url: str = "http://localhost:8088/v1/embeddings",
        embed_batch_size: int = DEFAULT_EMBED_BATCH_SIZE,
        callback_manager: Optional[CallbackManager] = None,
        **kwargs: Any,
    ):
        self._api_endpoint_url = api_endpoint_url

        super().__init__(
            model_name=model_name,
            embed_batch_size=embed_batch_size,
            callback_manager=callback_manager,
            **kwargs,
        )

    @classmethod
    def class_name(cls) -> str:
        return "NeMoEmbedding"

    def _get_embedding(self, texts: List[str], input_type: str) -> List[List[float]]:
        payload = json.dumps(
            {"input": texts, "model": self.model_name, "input_type": input_type}
        )
        headers = {"Content-Type": "application/json"}

        response = requests.request(
            "POST", self._api_endpoint_url, headers=headers, data=payload
        )

        try:
            response.raise_for_status()
        except requests.exceptions.HTTPError:
            raise Exception(
                f"Endpoint returned a non-successful status code: "
                f"{response.status_code} "
                f"Response text: {response.text}"
            )
        else:
            response = json.loads(response.text)
            return response["data"]

    async def _aget_embedding(
        self, session: Any, texts: List[str], input_type: str
    ) -> List[List[float]]:
        headers = {"Content-Type": "application/json"}

        async with session.post(
            self._api_endpoint_url,
            json={"input": texts, "model": self.model_name, "input_type": input_type},
            headers=headers,
        ) as response:
            response.raise_for_status()
            answer = await response.text()
            answer = json.loads(answer)
            return answer["data"]

    def _get_query_embedding(self, query: str) -> List[float]:
        """Get query embedding."""
        return self._get_embedding([query], input_type="query")[0]["embedding"]

    def _get_text_embedding(self, text: str) -> List[float]:
        """Get text embedding."""
        return self._get_embedding([text], input_type="passage")[0]["embedding"]

    def _get_text_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Get text embeddings."""
        return [
            embedding["embedding"]
            for embedding in self._get_embedding(texts, input_type="passage")
        ]

    async def _aget_query_embedding(self, query: str) -> List[float]:
        """Asynchronously get query embedding."""
        async with aiohttp.ClientSession() as session:
            embedding = await self._aget_embedding(session, [query], input_type="query")
            return embedding[0]["embedding"]

    async def _aget_text_embedding(self, text: str) -> List[float]:
        """Asynchronously get text embedding."""
        async with aiohttp.ClientSession() as session:
            embedding = await self._aget_embedding(
                session, [text], input_type="passage"
            )
            return embedding[0]["embedding"]

    async def _aget_text_embeddings(self, texts: List[str]) -> List[float]:
        """Asynchronously get text embedding."""
        async with aiohttp.ClientSession() as session:
            embeddings = await self._aget_embedding(
                session, texts, input_type="passage"
            )
            return [embedding["embedding"] for embedding in embeddings]
