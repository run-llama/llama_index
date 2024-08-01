from enum import Enum
from typing import Any, List, Optional

from llama_index.legacy.bridge.pydantic import Field, PrivateAttr
from llama_index.legacy.callbacks import CallbackManager
from llama_index.legacy.core.embeddings.base import BaseEmbedding


class NomicAITaskType(str, Enum):
    SEARCH_QUERY = "search_query"
    SEARCH_DOCUMENT = "search_document"
    CLUSTERING = "clustering"
    CLASSIFICATION = "classification"


TASK_TYPES = [
    NomicAITaskType.SEARCH_QUERY,
    NomicAITaskType.SEARCH_DOCUMENT,
    NomicAITaskType.CLUSTERING,
    NomicAITaskType.CLASSIFICATION,
]


class NomicEmbedding(BaseEmbedding):
    """NomicEmbedding uses the Nomic API to generate embeddings."""

    # Instance variables initialized via Pydantic's mechanism
    query_task_type: Optional[str] = Field(description="Query Embedding prefix")
    document_task_type: Optional[str] = Field(description="Document Embedding prefix")
    model_name: str = Field(description="Embedding model name")
    _model: Any = PrivateAttr()

    def __init__(
        self,
        model_name: str = "nomic-embed-text-v1",
        embed_batch_size: int = 32,
        api_key: Optional[str] = None,
        callback_manager: Optional[CallbackManager] = None,
        query_task_type: Optional[str] = "search_query",
        document_task_type: Optional[str] = "search_document",
        **kwargs: Any,
    ) -> None:
        if query_task_type not in TASK_TYPES or document_task_type not in TASK_TYPES:
            raise ValueError(
                f"Invalid task type {query_task_type}, {document_task_type}. Must be one of {TASK_TYPES}"
            )

        try:
            import nomic
            from nomic import embed
        except ImportError:
            raise ImportError(
                "NomicEmbedding requires the 'nomic' package to be installed.\n"
                "Please install it with `pip install nomic`."
            )

        if api_key is not None:
            nomic.cli.login(api_key)
        super().__init__(
            model_name=model_name,
            embed_batch_size=embed_batch_size,
            callback_manager=callback_manager,
            _model=embed,
            query_task_type=query_task_type,
            document_task_type=document_task_type,
            **kwargs,
        )
        self._model = embed
        self.model_name = model_name
        self.query_task_type = query_task_type
        self.document_task_type = document_task_type

    @classmethod
    def class_name(cls) -> str:
        return "NomicEmbedding"

    def _embed(
        self, texts: List[str], task_type: Optional[str] = None
    ) -> List[List[float]]:
        """Embed sentences using NomicAI."""
        result = self._model.text(texts, model=self.model_name, task_type=task_type)
        return result["embeddings"]

    def _get_query_embedding(self, query: str) -> List[float]:
        """Get query embedding."""
        return self._embed([query], task_type=self.query_task_type)[0]

    async def _aget_query_embedding(self, query: str) -> List[float]:
        """Get query embedding async."""
        return self._get_query_embedding(query)

    def _get_text_embedding(self, text: str) -> List[float]:
        """Get text embedding."""
        return self._embed([text], task_type=self.document_task_type)[0]

    async def _aget_text_embedding(self, text: str) -> List[float]:
        """Get text embedding async."""
        return self._get_text_embedding(text)

    def _get_text_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Get text embeddings."""
        return self._embed(texts, task_type=self.document_task_type)
