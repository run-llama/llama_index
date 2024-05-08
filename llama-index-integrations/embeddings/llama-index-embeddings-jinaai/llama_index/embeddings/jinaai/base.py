"""Jina embeddings file."""

from typing import Any, List, Optional

import requests
import numpy as np
from llama_index.core.base.embeddings.base import (
    DEFAULT_EMBED_BATCH_SIZE,
    BaseEmbedding,
)
from llama_index.core.bridge.pydantic import Field, PrivateAttr
from llama_index.core.callbacks.base import CallbackManager
from llama_index.core.base.llms.generic_utils import get_from_param_or_env

MAX_BATCH_SIZE = 2048

API_URL = "https://api.jina.ai/v1/embeddings"

VALID_ENCODING = ["float", "ubinary", "binary"]


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
    _encoding_queries: str = PrivateAttr()
    _encoding_documents: str = PrivateAttr()

    def __init__(
        self,
        model: str = "jina-embeddings-v2-base-en",
        embed_batch_size: int = DEFAULT_EMBED_BATCH_SIZE,
        api_key: Optional[str] = None,
        callback_manager: Optional[CallbackManager] = None,
        encoding_queries: Optional[str] = None,
        encoding_documents: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            embed_batch_size=embed_batch_size,
            callback_manager=callback_manager,
            model=model,
            api_key=api_key,
            **kwargs,
        )
        self._encoding_queries = encoding_queries or "float"
        self._encoding_documents = encoding_documents or "float"

        assert (
            self._encoding_documents in VALID_ENCODING
        ), f"Encoding Documents parameter {self._encoding_documents} not supported. Please choose one of {VALID_ENCODING}"
        assert (
            self._encoding_queries in VALID_ENCODING
        ), f"Encoding Queries parameter {self._encoding_documents} not supported. Please choose one of {VALID_ENCODING}"

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
        return self._get_embeddings([query], encoding_type=self._encoding_queries)[0]

    async def _aget_query_embedding(self, query: str) -> List[float]:
        """The asynchronous version of _get_query_embedding."""
        result = await self._aget_embeddings(
            [query], encoding_type=self._encoding_queries
        )
        return result[0]

    def _get_text_embedding(self, text: str) -> List[float]:
        """Get text embedding."""
        return self._get_text_embeddings([text])[0]

    async def _aget_text_embedding(self, text: str) -> List[float]:
        """Asynchronously get text embedding."""
        result = await self._aget_text_embeddings([text])
        return result[0]

    def _get_text_embeddings(self, texts: List[str]) -> List[List[float]]:
        return self._get_embeddings(texts=texts, encoding_type=self._encoding_documents)

    async def _aget_text_embeddings(
        self,
        texts: List[str],
    ) -> List[List[float]]:
        return await self._aget_embeddings(
            texts=texts, encoding_type=self._encoding_documents
        )

    def _get_embeddings(
        self, texts: List[str], encoding_type: str = "float"
    ) -> List[List[float]]:
        """Get embeddings."""
        # Call Jina AI Embedding API
        resp = self._session.post(  # type: ignore
            API_URL,
            json={"input": texts, "model": self.model, "encoding_type": encoding_type},
        ).json()
        if "data" not in resp:
            raise RuntimeError(resp["detail"])

        embeddings = resp["data"]

        # Sort resulting embeddings by index
        sorted_embeddings = sorted(embeddings, key=lambda e: e["index"])  # type: ignore

        # Return just the embeddings
        if encoding_type == "ubinary":
            return [
                np.unpackbits(np.array(result["embedding"], dtype="uint8")).tolist()
                for result in sorted_embeddings
            ]
        elif encoding_type == "binary":
            return [
                np.unpackbits(
                    (np.array(result["embedding"]) + 128).astype("uint8")
                ).tolist()
                for result in sorted_embeddings
            ]
        return [result["embedding"] for result in sorted_embeddings]

    async def _aget_embeddings(
        self, texts: List[str], encoding_type: str = "float"
    ) -> List[List[float]]:
        """Asynchronously get text embeddings."""
        import aiohttp

        async with aiohttp.ClientSession(trust_env=True) as session:
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Accept-Encoding": "identity",
            }
            async with session.post(
                f"{API_URL}",
                json={
                    "input": texts,
                    "model": self.model,
                    "encoding_type": encoding_type,
                },
                headers=headers,
            ) as response:
                resp = await response.json()
                response.raise_for_status()
                embeddings = resp["data"]

                # Sort resulting embeddings by index
                sorted_embeddings = sorted(embeddings, key=lambda e: e["index"])  # type: ignore

                # Return just the embeddings
                if encoding_type == "ubinary":
                    return [
                        np.unpackbits(
                            np.array(result["embedding"], dtype="uint8")
                        ).tolist()
                        for result in sorted_embeddings
                    ]
                elif encoding_type == "binary":
                    return [
                        np.unpackbits(
                            (np.array(result["embedding"]) + 128).astype("uint8")
                        ).tolist()
                        for result in sorted_embeddings
                    ]
                return [result["embedding"] for result in sorted_embeddings]
