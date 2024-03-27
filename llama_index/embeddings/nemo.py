"""NeMo embeddings file."""

import json
import requests
from typing import Any, List, Optional

from llama_index.core.base.embeddings.base import (
    DEFAULT_EMBED_BATCH_SIZE,
    BaseEmbedding,
)
from llama_index.core.callbacks.base import CallbackManager


class NemoEmbedding(BaseEmbedding):
    """Nvidia NeMo embeddings.
    """

    def __init__(
        self,
        model_name: str = "NV-Embed-QA-003",
        api_endpoint_url: str = "http://localhost:8088/v1/embeddings",
        embed_batch_size: int = DEFAULT_EMBED_BATCH_SIZE,
        callback_manager: Optional[CallbackManager] = None,
        **kwargs: Any,
    ):
        self.api_endpoint_url = api_endpoint_url

        super().__init__(
            model_name=model_name,
            embed_batch_size=embed_batch_size,
            callback_manager=callback_manager,
            **kwargs,
        )

    @classmethod
    def class_name(cls) -> str:
        return "NemoEmbedding"

    def _get_embedding(self, text: str, input_type: str):
        payload = json.dumps({
            "input": text,
            "model": self.model_name,
            "input_type": input_type
        })
        headers = {
            'Content-Type': 'application/json'
        }

        response = requests.request(
            "POST", self.api_endpoint_url, headers=headers, data=payload)
        response = json.loads(response.text)

        return response["data"][0]["embedding"]

    def _get_query_embedding(self, query: str) -> List[float]:
        return self._get_embedding(text, input_type="query")
        
    def _get_text_embedding(self, text: str) -> List[float]:
        return self._get_embedding(text, input_type="passage")

    def _get_text_embeddings(self, texts: List[str]) -> List[List[float]]:
        return [self._get_embedding(text, input_type="passage") for text in texts]
