"""Google PaLM embeddings file."""

import deprecated
from typing import Any, List, Optional

from llama_index.core.base.embeddings.base import (
    DEFAULT_EMBED_BATCH_SIZE,
    BaseEmbedding,
)
from llama_index.core.bridge.pydantic import PrivateAttr
from llama_index.core.callbacks.base import CallbackManager

import google.generativeai as palm


@deprecated.deprecated(
    reason=(
        "Should use `llama-index-embeddings-google-genai` instead, using Google's latest unified SDK. "
        "See: https://docs.llamaindex.ai/en/stable/examples/embeddings/google_genai/"
    )
)
class GooglePaLMEmbedding(BaseEmbedding):
    """Class for Google PaLM embeddings.

    Args:
        model_name (str): Model for embedding.
            Defaults to "models/embedding-gecko-001".

        api_key (Optional[str]): API key to access the model. Defaults to None.
    """

    _model: Any = PrivateAttr()

    def __init__(
        self,
        model_name: str = "models/embedding-gecko-001",
        api_key: Optional[str] = None,
        embed_batch_size: int = DEFAULT_EMBED_BATCH_SIZE,
        callback_manager: Optional[CallbackManager] = None,
        **kwargs: Any,
    ):
        super().__init__(
            model_name=model_name,
            embed_batch_size=embed_batch_size,
            callback_manager=callback_manager,
            **kwargs,
        )
        palm.configure(api_key=api_key)
        self._model = palm

    @classmethod
    def class_name(cls) -> str:
        return "PaLMEmbedding"

    def _get_query_embedding(self, query: str) -> List[float]:
        """Get query embedding."""
        return self._model.generate_embeddings(model=self.model_name, text=query)[
            "embedding"
        ]

    async def _aget_query_embedding(self, query: str) -> List[float]:
        """The asynchronous version of _get_query_embedding."""
        return await self._model.aget_embedding(query)

    def _get_text_embedding(self, text: str) -> List[float]:
        """Get text embedding."""
        return self._model.generate_embeddings(model=self.model_name, text=text)[
            "embedding"
        ]

    async def _aget_text_embedding(self, text: str) -> List[float]:
        """Asynchronously get text embedding."""
        return self._model._get_text_embedding(text)

    def _get_text_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Get text embeddings."""
        return self._model.generate_embeddings(model=self.model_name, text=texts)[
            "embedding"
        ]

    async def _aget_text_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Asynchronously get text embeddings."""
        return await self._model._get_embeddings(texts)
