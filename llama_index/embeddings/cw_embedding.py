from typing import Optional, Any, List

from llama_index.bridge.pydantic import Field
from llama_index.callbacks import CallbackManager
from llama_index.embeddings import BaseEmbedding
from llama_index.embeddings.base import Embedding, DEFAULT_EMBED_BATCH_SIZE


class CwEmbedding(BaseEmbedding):
    url: str = Field(description="The Cloudwalk Embedding url.")
    model_name: str = Field(description="The Cloudwalk Embedding model name.")
    timeout: float = Field(description="Timeout in seconds for the request.", default=60.0)

    def __init__(
            self,
            url: str,
            model_name: str,
            timeout: float = 60.0,
            embed_batch_size: int = DEFAULT_EMBED_BATCH_SIZE,
            callback_manager: Optional[CallbackManager] = None,
            **kwargs: Any,
    ) -> None:
        super().__init__(
            embed_batch_size=embed_batch_size,
            callback_manager=callback_manager,
            model_name=model_name,
            url=url,
            timeout=timeout,
            **kwargs,
        )

    @classmethod
    def class_name(cls) -> str:
        return "CwEmbedding"

    def _call_api(self, texts: str) -> List[float]:
        import httpx

        headers = {"Content-Type": "application/json"}
        json_data = {"input": texts, "model": self.model_name}

        with httpx.Client() as client:
            response = client.post(
                self.url,
                headers=headers,
                json=json_data,
                timeout=self.timeout,
            )

        return response.json()["data"][0]["embedding"]

    async def _acall_api(self, texts: str) -> List[float]:
        import httpx

        headers = {"Content-Type": "application/json"}
        json_data = {"input": texts, "model": self.model_name}

        async with httpx.AsyncClient() as client:
            response = await client.post(
                self.url,
                headers=headers,
                json=json_data,
                timeout=self.timeout,
            )

        return response.json()["data"][0]["embedding"]

    def _get_query_embedding(self, query: str) -> Embedding:
        return self._call_api(query)

    async def _aget_query_embedding(self, query: str) -> Embedding:
        return await self._acall_api(query)

    def _get_text_embedding(self, text: str) -> Embedding:
        return self._call_api(text)
