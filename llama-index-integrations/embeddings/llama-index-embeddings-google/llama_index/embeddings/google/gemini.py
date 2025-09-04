"""Gemini embeddings file."""

import deprecated
from typing import Any, List, Optional

from llama_index.core.base.embeddings.base import (
    DEFAULT_EMBED_BATCH_SIZE,
    BaseEmbedding,
)
from llama_index.core.bridge.pydantic import Field, PrivateAttr
from llama_index.core.callbacks.base import CallbackManager

import google.generativeai as gemini


@deprecated.deprecated(
    reason=(
        "Should use `llama-index-embeddings-google-genai` instead, using Google's latest unified SDK. "
        "See: https://docs.llamaindex.ai/en/stable/examples/embeddings/google_genai/"
    )
)
class GeminiEmbedding(BaseEmbedding):
    """
    Google Gemini embeddings.

    Args:
        model_name (str): Model for embedding.
            Defaults to "models/embedding-001".

        api_key (Optional[str]): API key to access the model. Defaults to None.

    """

    _model: Any = PrivateAttr()
    title: Optional[str] = Field(
        default="",
        description="Title is only applicable for retrieval_document tasks, and is used to represent a document title. For other tasks, title is invalid.",
    )
    task_type: Optional[str] = Field(
        default="retrieval_document",
        description="The task for embedding model.",
    )

    def __init__(
        self,
        model_name: str = "models/embedding-001",
        task_type: Optional[str] = "retrieval_document",
        api_key: Optional[str] = None,
        title: Optional[str] = None,
        embed_batch_size: int = DEFAULT_EMBED_BATCH_SIZE,
        callback_manager: Optional[CallbackManager] = None,
        **kwargs: Any,
    ):
        super().__init__(
            model_name=model_name,
            embed_batch_size=embed_batch_size,
            callback_manager=callback_manager,
            title=title,
            task_type=task_type,
            **kwargs,
        )
        gemini.configure(api_key=api_key)
        self._model = gemini

    @classmethod
    def class_name(cls) -> str:
        return "GeminiEmbedding"

    def _get_query_embedding(self, query: str) -> List[float]:
        """Get query embedding."""
        return self._model.embed_content(
            model=self.model_name,
            content=query,
            title=self.title,
            task_type=self.task_type,
        )["embedding"]

    def _get_text_embedding(self, text: str) -> List[float]:
        """Get text embedding."""
        return self._model.embed_content(
            model=self.model_name,
            content=text,
            title=self.title,
            task_type=self.task_type,
        )["embedding"]

    def _get_text_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Get text embeddings."""
        return [
            self._model.embed_content(
                model=self.model_name,
                content=text,
                title=self.title,
                task_type=self.task_type,
            )["embedding"]
            for text in texts
        ]

    ### Async methods ###
    # need to wait async calls from Gemini side to be implemented.
    # Issue: https://github.com/google/generative-ai-python/issues/125
    async def _aget_query_embedding(self, query: str) -> List[float]:
        """The asynchronous version of _get_query_embedding."""
        return self._get_query_embedding(query)

    async def _aget_text_embedding(self, text: str) -> List[float]:
        """Asynchronously get text embedding."""
        return self._get_text_embedding(text)

    async def _aget_text_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Asynchronously get text embeddings."""
        return self._get_text_embeddings(texts)
