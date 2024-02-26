"""Gemini embeddings file."""

from typing import Any, List, Optional

from llama_index.legacy.bridge.pydantic import Field, PrivateAttr
from llama_index.legacy.callbacks.base import CallbackManager
from llama_index.legacy.core.embeddings.base import (
    DEFAULT_EMBED_BATCH_SIZE,
    BaseEmbedding,
)


class GeminiEmbedding(BaseEmbedding):
    """Google Gemini embeddings.

    Args:
        model_name (str): Model for embedding.
            Defaults to "models/embedding-001".

        api_key (Optional[str]): API key to access the model. Defaults to None.
        api_base (Optional[str]): API base to access the model. Defaults to Official Base.
        transport (Optional[str]): Transport to access the model.
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
        api_base: Optional[str] = None,
        transport: Optional[str] = None,
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
        # API keys are optional. The API can be authorised via OAuth (detected
        # environmentally) or by the GOOGLE_API_KEY environment variable.
        config_params: Dict[str, Any] = {
            "api_key": api_key or os.getenv("GOOGLE_API_KEY"),
        }
        if api_base:
            config_params["client_options"] = {"api_endpoint": api_base}
        if transport:
            config_params["transport"] = transport
        # transport: A string, one of: [`rest`, `grpc`, `grpc_asyncio`].
        gemini.configure(**config_params)
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
