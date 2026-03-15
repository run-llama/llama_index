"""GPU-Bridge embeddings for LlamaIndex."""

import time
from typing import Any, List, Optional

import requests
from llama_index.core.base.embeddings.base import DEFAULT_EMBED_BATCH_SIZE, BaseEmbedding
from llama_index.core.bridge.pydantic import Field
from llama_index.core.callbacks.base import CallbackManager

GPUBRIDGE_API_URL = "https://api.gpubridge.xyz/run"
GPUBRIDGE_BASE_URL = "https://api.gpubridge.xyz"


class GPUBridgeEmbedding(BaseEmbedding):
    """GPU-Bridge text embeddings.

    GPU-Bridge provides high-throughput embedding inference at ~$0.00002/call
    via a single `POST /run` endpoint. Supports API key auth and x402 USDC payments.

    Install: ``pip install llama-index-embeddings-gpubridge``

    .. code-block:: python

        from llama_index.embeddings.gpubridge import GPUBridgeEmbedding

        embed_model = GPUBridgeEmbedding(api_key="gpub_...")

    """

    api_key: Optional[str] = Field(
        default=None,
        description="GPU-Bridge API key. Register at https://gpubridge.xyz",
    )
    service: str = Field(
        default="embedding-l4",
        description="GPU-Bridge embedding service.",
    )
    base_url: str = Field(
        default=GPUBRIDGE_API_URL,
        description="GPU-Bridge API endpoint.",
    )

    def __init__(
        self,
        api_key: Optional[str] = None,
        service: str = "embedding-l4",
        embed_batch_size: int = DEFAULT_EMBED_BATCH_SIZE,
        callback_manager: Optional[CallbackManager] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            api_key=api_key,
            service=service,
            embed_batch_size=embed_batch_size,
            callback_manager=callback_manager,
            **kwargs,
        )

    @classmethod
    def class_name(cls) -> str:
        return "GPUBridgeEmbedding"

    def _get_headers(self) -> dict:
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        return headers

    def _poll_job(self, status_url: str) -> dict:
        for _ in range(30):
            time.sleep(1)
            resp = requests.get(
                f"{GPUBRIDGE_BASE_URL}{status_url}",
                headers=self._get_headers(),
                timeout=10,
            )
            resp.raise_for_status()
            data = resp.json()
            if data.get("status") == "completed":
                return data
            if data.get("status") == "failed":
                raise ValueError(f"GPU-Bridge job failed: {data}")
        raise TimeoutError("GPU-Bridge embedding job timed out")

    def _embed_text(self, text: str) -> List[float]:
        payload = {"service": self.service, "input": {"texts": [text]}}
        resp = requests.post(
            self.base_url,
            json=payload,
            headers=self._get_headers(),
            timeout=30,
        )
        resp.raise_for_status()
        data = resp.json()

        if "error" in data:
            raise ValueError(f"GPU-Bridge error: {data['error']}")

        if data.get("status") == "pending" and "status_url" in data:
            data = self._poll_job(data["status_url"])

        output = data.get("output", {})
        return output.get("embedding", [])

    def _get_query_embedding(self, query: str) -> List[float]:
        return self._embed_text(query)

    def _get_text_embedding(self, text: str) -> List[float]:
        return self._embed_text(text)

    def _get_text_embeddings(self, texts: List[str]) -> List[List[float]]:
        return [self._embed_text(t) for t in texts]

    async def _aget_query_embedding(self, query: str) -> List[float]:
        return self._get_query_embedding(query)

    async def _aget_text_embedding(self, text: str) -> List[float]:
        return self._get_text_embedding(text)
