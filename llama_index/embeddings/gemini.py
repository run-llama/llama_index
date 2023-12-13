"""Gemini embeddings file."""

from typing import Any, List, Optional

from llama_index.bridge.pydantic import PrivateAttr
from llama_index.callbacks.base import CallbackManager
from llama_index.embeddings.base import DEFAULT_EMBED_BATCH_SIZE, BaseEmbedding


class GeminiEmbedding(BaseEmbedding):
    """Google Gemini embeddings.

    Args:
        model_name (str): Model for embedding.
            Defaults to "models/embedding-001".

        api_key (Optional[str]): API key to access the model. Defaults to None.
    """

    _model: Any = PrivateAttr()

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
        try:
            import google.generativeai as gemini
        except ImportError:
            raise ImportError(
                "google-generativeai package not found, install with"
                "'pip install google-generativeai'"
            )
        gemini.configure(api_key=api_key)
        self._model = gemini

        super().__init__(
            model_name=model_name,
            embed_batch_size=embed_batch_size,
            callback_manager=callback_manager,
            **kwargs,
        )
        self.title = title
        self.task_type = task_type

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

    async def _aget_query_embedding(self, query: str) -> List[float]:
        """The asynchronous version of _get_query_embedding."""
        return await self._model.aget_embedding(query)

    def _get_text_embedding(self, text: str) -> List[float]:
        """Get text embedding."""
        return self._model.embed_content(
            model=self.model_name,
            content=text,
            title=self.title,
            task_type=self.task_type,
        )["embedding"]

    async def _aget_text_embedding(self, text: str) -> List[float]:
        """Asynchronously get text embedding."""
        raise NotImplementedError

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

    async def _aget_text_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Asynchronously get text embeddings."""
        raise NotImplementedError
