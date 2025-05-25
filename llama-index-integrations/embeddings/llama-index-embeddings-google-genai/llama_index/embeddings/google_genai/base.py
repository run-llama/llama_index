"""Gemini embeddings file."""

import os
from typing import Any, Dict, List, Optional, TypedDict

from llama_index.core.base.embeddings.base import (
    DEFAULT_EMBED_BATCH_SIZE,
    BaseEmbedding,
)
from llama_index.core.bridge.pydantic import Field, PrivateAttr
from llama_index.core.callbacks.base import CallbackManager

import google.genai
import google.auth.credentials
import google.genai.types as types


class VertexAIConfig(TypedDict):
    credentials: Optional[google.auth.credentials.Credentials] = None
    project: Optional[str] = None
    location: Optional[str] = None


class GoogleGenAIEmbedding(BaseEmbedding):
    """
    Google GenAI embeddings.

    Args:
        model_name (str): Model for embedding.
            Defaults to "text-embedding-005".
        api_key (Optional[str]): API key to access the model. Defaults to None.
        embedding_config (Optional[types.EmbedContentConfigOrDict]): Embedding config to access the model. Defaults to None.
        vertexai_config (Optional[VertexAIConfig]): Vertex AI config to access the model. Defaults to None.
        http_options (Optional[types.HttpOptions]): HTTP options to access the model. Defaults to None.
        debug_config (Optional[google.genai.client.DebugConfig]): Debug config to access the model. Defaults to None.
        embed_batch_size (int): Batch size for embedding. Defaults to 100.
        callback_manager (Optional[CallbackManager]): Callback manager to access the model. Defaults to None.

    Examples:
        `pip install llama-index-embeddings-google-genai`

        ```python
        from llama_index.embeddings.google_genai import GoogleGenAIEmbedding

        embed_model = GoogleGenAIEmbedding(model_name="text-embedding-005", api_key="...")
        ```

    """

    _client: google.genai.Client = PrivateAttr()
    _embedding_config: types.EmbedContentConfigOrDict = PrivateAttr()

    embedding_config: Optional[types.EmbedContentConfigOrDict] = Field(
        default=None, description="""Used to override embedding config."""
    )

    def __init__(
        self,
        model_name: str = "text-embedding-004",
        api_key: Optional[str] = None,
        embedding_config: Optional[types.EmbedContentConfigOrDict] = None,
        vertexai_config: Optional[VertexAIConfig] = None,
        http_options: Optional[types.HttpOptions] = None,
        debug_config: Optional[google.genai.client.DebugConfig] = None,
        embed_batch_size: int = DEFAULT_EMBED_BATCH_SIZE,
        callback_manager: Optional[CallbackManager] = None,
        **kwargs: Any,
    ):
        super().__init__(
            model_name=model_name,
            embedding_config=embedding_config,
            embed_batch_size=embed_batch_size,
            callback_manager=callback_manager,
            **kwargs,
        )

        # API keys are optional. The API can be authorised via OAuth (detected
        # environmentally) or by the GOOGLE_API_KEY environment variable.
        api_key = api_key or os.getenv("GOOGLE_API_KEY", None)
        vertexai = vertexai_config is not None or os.getenv(
            "GOOGLE_GENAI_USE_VERTEXAI", False
        )
        project = (vertexai_config or {}).get("project") or os.getenv(
            "GOOGLE_CLOUD_PROJECT", None
        )
        location = (vertexai_config or {}).get("location") or os.getenv(
            "GOOGLE_CLOUD_LOCATION", None
        )

        config_params: Dict[str, Any] = {
            "api_key": api_key,
        }

        if vertexai_config is not None:
            config_params.update(vertexai_config)
            config_params["api_key"] = None
            config_params["vertexai"] = True
        elif vertexai:
            config_params["project"] = project
            config_params["location"] = location
            config_params["api_key"] = None
            config_params["vertexai"] = True

        if http_options:
            config_params["http_options"] = http_options

        if debug_config:
            config_params["debug_config"] = debug_config

        self._client = google.genai.Client(**config_params)

    @classmethod
    def class_name(cls) -> str:
        return "GeminiEmbedding"

    def _embed_texts(
        self, texts: List[str], task_type: Optional[str] = None
    ) -> List[List[float]]:
        """Embed texts."""
        # Set the task type if it is not already set
        if task_type and not self.embedding_config:
            self.embedding_config = types.EmbedContentConfig(task_type=task_type)

        results = self._client.models.embed_content(
            model=self.model_name,
            contents=texts,
            config=self.embedding_config,
        )
        return [result.values for result in results.embeddings]

    async def _aembed_texts(
        self, texts: List[str], task_type: Optional[str] = None
    ) -> List[List[float]]:
        """Asynchronously embed texts."""
        # Set the task type if it is not already set
        if task_type and not self.embedding_config:
            self.embedding_config = types.EmbedContentConfig(task_type=task_type)

        results = await self._client.aio.models.embed_content(
            model=self.model_name,
            contents=texts,
            config=self.embedding_config,
        )
        return [result.values for result in results.embeddings]

    def _get_query_embedding(self, query: str) -> List[float]:
        """Get query embedding."""
        return self._embed_texts([query], task_type="RETRIEVAL_QUERY")[0]

    def _get_text_embedding(self, text: str) -> List[float]:
        """Get text embedding."""
        return self._embed_texts([text], task_type="RETRIEVAL_DOCUMENT")[0]

    def _get_text_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Get text embeddings."""
        return self._embed_texts(texts, task_type="RETRIEVAL_DOCUMENT")

    async def _aget_query_embedding(self, query: str) -> List[float]:
        """The asynchronous version of _get_query_embedding."""
        return (await self._aembed_texts([query], task_type="RETRIEVAL_QUERY"))[0]

    async def _aget_text_embedding(self, text: str) -> List[float]:
        """Asynchronously get text embedding."""
        return (await self._aembed_texts([text], task_type="RETRIEVAL_DOCUMENT"))[0]

    async def _aget_text_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Asynchronously get text embeddings."""
        return await self._aembed_texts(texts, task_type="RETRIEVAL_DOCUMENT")
