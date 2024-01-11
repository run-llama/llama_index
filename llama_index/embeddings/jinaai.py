"""Jina embeddings file."""

from typing import Any, List, Optional

import requests

from llama_index.bridge.pydantic import Field, PrivateAttr
from llama_index.callbacks.base import CallbackManager
from llama_index.core.embeddings.base import DEFAULT_EMBED_BATCH_SIZE, BaseEmbedding
from llama_index.llms.generic_utils import get_from_param_or_env

MAX_BATCH_SIZE = 2048

API_URL = "https://api.jina.ai/v1/embeddings"


class JinaEmbedding(BaseEmbedding):
    """JinaAI class for embeddings.

    Args:
        model (str): Model for embedding.
            Defaults to `jina-embeddings-v2-base-en`
    """

    api_key: str = Field(default=None, description="The JinaAI API key.")
    model: str = Field(
        default="jina-embeddings-v2-base-en",
        description="The model to use when calling Jina AI API",
    )

    _session: Any = PrivateAttr()

    def __init__(
        self,
        model: str = "jina-embeddings-v2-base-en",
        embed_batch_size: int = DEFAULT_EMBED_BATCH_SIZE,
        api_key: Optional[str] = None,
        callback_manager: Optional[CallbackManager] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            embed_batch_size=embed_batch_size,
            callback_manager=callback_manager,
            model=model,
            api_key=api_key,
            **kwargs,
        )
        self.api_key = get_from_param_or_env("api_key", api_key, "JINAAI_API_KEY", "")
        self.model = model
        self._session = requests.Session()
        self._session.headers.update(
            {"Authorization": f"Bearer {api_key}", "Accept-Encoding": "identity"}
        )

    @classmethod
    def class_name(cls) -> str:
        return "JinaAIEmbedding"

    def _get_query_embedding(self, query: str) -> List[float]:
        """Get query embedding."""
        return self._get_text_embedding(query)

    async def _aget_query_embedding(self, query: str) -> List[float]:
        """The asynchronous version of _get_query_embedding."""
        return await self._aget_text_embedding(query)

    def _get_text_embedding(self, text: str) -> List[float]:
        """Get text embedding."""
        return self._get_text_embeddings([text])[0]

    async def _aget_text_embedding(self, text: str) -> List[float]:
        """Asynchronously get text embedding."""
        result = await self._aget_text_embeddings([text])
        return result[0]

    def _get_text_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Get text embeddings."""
        # Call Jina AI Embedding API
        resp = self._session.post(  # type: ignore
            API_URL, json={"input": texts, "model": self.model}
        ).json()
        if "data" not in resp:
            raise RuntimeError(resp["detail"])

        embeddings = resp["data"]

        # Sort resulting embeddings by index
        sorted_embeddings = sorted(embeddings, key=lambda e: e["index"])  # type: ignore

        # Return just the embeddings
        return [result["embedding"] for result in sorted_embeddings]

    async def _aget_text_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Asynchronously get text embeddings."""
        import aiohttp

        async with aiohttp.ClientSession(trust_env=True) as session:
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Accept-Encoding": "identity",
            }
            async with session.post(
                f"{API_URL}",
                json={"input": texts, "model": self.model},
                headers=headers,
            ) as response:
                resp = await response.json()
                response.raise_for_status()
                embeddings = resp["data"]

                # Sort resulting embeddings by index
                sorted_embeddings = sorted(embeddings, key=lambda e: e["index"])  # type: ignore

                # Return just the embeddings
                return [result["embedding"] for result in sorted_embeddings]
