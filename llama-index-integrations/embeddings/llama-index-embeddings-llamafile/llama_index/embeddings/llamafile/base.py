from typing import Optional, List

import httpx
from httpx import Timeout

from llama_index.core.base.embeddings.base import BaseEmbedding, Embedding
from llama_index.core.bridge.pydantic import Field
from llama_index.core.callbacks.base import CallbackManager

DEFAULT_REQUEST_TIMEOUT = 30.0


class LlamafileEmbedding(BaseEmbedding):
    """
    Class for llamafile embeddings.

    llamafile lets you distribute and run large language models with a
    single file.

    To get started, see: https://github.com/Mozilla-Ocho/llamafile

    To use this class, you will need to first:

    1. Download a llamafile.
    2. Make the downloaded file executable: `chmod +x path/to/model.llamafile`
    3. Start the llamafile in server mode with embeddings enabled:

        `./path/to/model.llamafile --server --nobrowser --embedding`

    """

    base_url: str = Field(
        description="base url of the llamafile server", default="http://localhost:8080"
    )

    request_timeout: float = Field(
        default=DEFAULT_REQUEST_TIMEOUT,
        description="The timeout for making http request to llamafile API server",
    )

    def __init__(
        self,
        base_url: str = "http://localhost:8080",
        callback_manager: Optional[CallbackManager] = None,
        **kwargs,
    ) -> None:
        super().__init__(
            base_url=base_url,
            callback_manager=callback_manager or CallbackManager([]),
            **kwargs,
        )

    @classmethod
    def class_name(cls) -> str:
        return "LlamafileEmbedding"

    def _get_query_embedding(self, query: str) -> Embedding:
        return self._get_text_embedding(query)

    async def _aget_query_embedding(self, query: str) -> Embedding:
        return await self._aget_text_embedding(query)

    def _get_text_embedding(self, text: str) -> Embedding:
        """
        Embed the input text synchronously.
        """
        request_body = {
            "content": text,
        }

        with httpx.Client(timeout=Timeout(self.request_timeout)) as client:
            response = client.post(
                url=f"{self.base_url}/embedding",
                headers={"Content-Type": "application/json"},
                json=request_body,
            )
            response.encoding = "utf-8"
            response.raise_for_status()

            return response.json()["embedding"]

    async def _aget_text_embedding(self, text: str) -> Embedding:
        """
        Embed the input text asynchronously.
        """
        request_body = {
            "content": text,
        }

        async with httpx.AsyncClient(timeout=Timeout(self.request_timeout)) as client:
            response = await client.post(
                url=f"{self.base_url}/embedding",
                headers={"Content-Type": "application/json"},
                json=request_body,
            )
            response.encoding = "utf-8"
            response.raise_for_status()

            return response.json()["embedding"]

    def _get_text_embeddings(self, texts: List[str]) -> List[Embedding]:
        """
        Embed the input texts synchronously.
        """
        request_body = {
            "content": texts,
        }

        with httpx.Client(timeout=Timeout(self.request_timeout)) as client:
            response = client.post(
                url=f"{self.base_url}/embedding",
                headers={"Content-Type": "application/json"},
                json=request_body,
            )
            response.encoding = "utf-8"
            response.raise_for_status()

            return [output["embedding"] for output in response.json()["results"]]

    async def _aget_text_embeddings(self, texts: List[str]) -> Embedding:
        """
        Embed the input text asynchronously.
        """
        request_body = {
            "content": texts,
        }

        async with httpx.AsyncClient(timeout=Timeout(self.request_timeout)) as client:
            response = await client.post(
                url=f"{self.base_url}/embedding",
                headers={"Content-Type": "application/json"},
                json=request_body,
            )
            response.encoding = "utf-8"
            response.raise_for_status()

            return [output["embedding"] for output in response.json()["results"]]
