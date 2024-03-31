"""Solar embeddings file."""

import json
import requests
from typing import Any, List, Optional

from llama_index.core.base.embeddings.base import (
    DEFAULT_EMBED_BATCH_SIZE,
    BaseEmbedding,
)
from llama_index.core.bridge.pydantic import Field
from llama_index.core.callbacks.base import CallbackManager
from llama_index.core.base.llms.generic_utils import get_from_param_or_env

DEFAULT_SOLAR_API_BASE = "https://api.upstage.ai/v1/solar/embeddings"


class SolarEmbedding(BaseEmbedding):
    """Solar embeddings."""

    api_key: str = Field(description="The Solar API key.")
    api_base: str = Field(
        default=DEFAULT_SOLAR_API_BASE,
        description="The base URL for Solar embedding API.",
    )

    def __init__(
        self,
        model_name: str = "solar-1-mini-embedding-query",
        api_key: str = "",
        api_base: str = DEFAULT_SOLAR_API_BASE,
        embed_batch_size: int = DEFAULT_EMBED_BATCH_SIZE,
        callback_manager: Optional[CallbackManager] = None,
        **kwargs: Any,
    ):
        api_key = get_from_param_or_env("api_key", api_key, "SOLAR_API_KEY", "")
        super().__init__(
            model_name=model_name,
            api_key=api_key,
            api_base=api_base,
            embed_batch_size=embed_batch_size,
            callback_manager=callback_manager,
            **kwargs,
        )

    @classmethod
    def class_name(cls) -> str:
        return "SolarEmbedding"

    def _get_embedding(self, text: str) -> List[float]:
        payload = json.dumps({"input": text, "model": self.model_name})
        headers = {
            "Content-Type": "application/json",
            "Authorization": "Bearer " + self.api_key,
        }

        response = requests.request(
            "POST", self.api_base, headers=headers, data=payload
        )
        if response.ok:
            response = json.loads(response.text)
            return response["data"][0]["embedding"]
        else:
            raise Exception(
                f"Failed to get {self.api_base} (status: {response.status_code})"
            )

    async def _aget_embedding(
        self, session: Any, text: str, input_type: str
    ) -> List[float]:
        headers = {"Content-Type": "application/json"}

        async with session.post(
            self.api_base,
            json={"input": text, "model": self.model},
            headers=headers,
        ) as response:
            response.raise_for_status()
            answer = await response.text()
            answer = json.loads(answer)
            return answer["data"][0]["embedding"]

    def _get_query_embedding(self, query: str) -> List[float]:
        return self._get_embedding(query, input_type="query")

    def _get_text_embedding(self, text: str) -> List[float]:
        return self._get_embedding(text)

    def _get_text_embeddings(self, texts: List[str]) -> List[List[float]]:
        return [self._get_embedding(text) for text in texts]

    async def _aget_query_embedding(self, query: str) -> List[float]:
        """Get query embedding async. For query embeddings, input_type='search_query'."""
        return self._get_query_embedding(query)

    async def _aget_text_embedding(self, text: str) -> List[float]:
        """Get text embedding async."""
        return self._get_text_embedding(text)
