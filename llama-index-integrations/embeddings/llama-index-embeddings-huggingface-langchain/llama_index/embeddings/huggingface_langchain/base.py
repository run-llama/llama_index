"""LlamaIndex embedding integration using langchain-huggingface.

Provides LlamaIndex-compatible embeddings using langchain-huggingface's
HuggingFaceEmbeddings (local sentence-transformers) and
HuggingFaceEndpointEmbeddings (remote via Inference API).
"""

import logging
import os
from typing import Any, List, Optional

from llama_index.core.base.embeddings.base import (
    DEFAULT_EMBED_BATCH_SIZE,
    BaseEmbedding,
)
from llama_index.core.bridge.pydantic import Field, PrivateAttr
from llama_index.core.callbacks import CallbackManager

logger = logging.getLogger(__name__)


class HuggingFaceLangChainEmbedding(BaseEmbedding):
    """LlamaIndex embeddings powered by langchain-huggingface.

    Supports both local embedding models (via sentence-transformers / HuggingFaceEmbeddings)
    and remote embedding models (via HuggingFace Inference API / HuggingFaceEndpointEmbeddings).

    Examples:
        `pip install llama-index-embeddings-huggingface-langchain`

        ```python
        from llama_index.embeddings.huggingface_langchain import (
            HuggingFaceLangChainEmbedding,
        )

        # Local embeddings (sentence-transformers)
        embed_model = HuggingFaceLangChainEmbedding(
            model_name="sentence-transformers/all-mpnet-base-v2",
        )
        embedding = embed_model.get_text_embedding("Hello world")
        print(f"Embedding dimensions: {len(embedding)}")

        # Remote embeddings via HuggingFace Inference API
        embed_model_api = HuggingFaceLangChainEmbedding(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            backend="api",
        )
        ```
    """

    backend: str = Field(
        default="local",
        description="'local' for sentence-transformers, 'api' for HuggingFace Inference API.",
    )
    huggingfacehub_api_token: Optional[str] = Field(
        default=None,
        description="HuggingFace API token. Falls back to HF_TOKEN or HUGGINGFACE_TOKEN env vars.",
    )
    encode_kwargs: dict = Field(
        default_factory=dict,
        description="Additional keyword arguments for the encode method (e.g. normalize_embeddings).",
    )
    model_init_kwargs: dict = Field(
        default_factory=dict,
        description="Additional keyword arguments for model initialization.",
    )

    _embedding_model: Any = PrivateAttr()

    def __init__(
        self,
        model_name: str = "sentence-transformers/all-mpnet-base-v2",
        backend: str = "local",
        huggingfacehub_api_token: Optional[str] = None,
        encode_kwargs: Optional[dict] = None,
        model_init_kwargs: Optional[dict] = None,
        embed_batch_size: int = DEFAULT_EMBED_BATCH_SIZE,
        callback_manager: Optional[CallbackManager] = None,
    ) -> None:
        encode_kwargs = encode_kwargs or {}
        model_init_kwargs = model_init_kwargs or {}

        super().__init__(
            model_name=model_name,
            backend=backend,
            huggingfacehub_api_token=huggingfacehub_api_token,
            encode_kwargs=encode_kwargs,
            model_init_kwargs=model_init_kwargs,
            embed_batch_size=embed_batch_size,
            callback_manager=callback_manager,
        )

        token = huggingfacehub_api_token
        if token is None:
            token = os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_TOKEN")

        if backend == "local":
            from langchain_huggingface import HuggingFaceEmbeddings

            self._embedding_model = HuggingFaceEmbeddings(
                model_name=model_name,
                encode_kwargs=encode_kwargs,
                **model_init_kwargs,
            )
        elif backend == "api":
            from langchain_huggingface import HuggingFaceEndpointEmbeddings

            self._embedding_model = HuggingFaceEndpointEmbeddings(
                model=model_name,
                huggingfacehub_api_token=token,
                **model_init_kwargs,
            )
        else:
            raise ValueError(
                f"Unknown backend '{backend}'. Supported values: 'local', 'api'."
            )

    @classmethod
    def class_name(cls) -> str:
        return "HuggingFaceLangChainEmbedding"

    def _get_query_embedding(self, query: str) -> List[float]:
        """Get query embedding."""
        return self._embedding_model.embed_query(query)

    async def _aget_query_embedding(self, query: str) -> List[float]:
        """Get query embedding (async)."""
        try:
            return await self._embedding_model.aembed_query(query)
        except NotImplementedError:
            return self._get_query_embedding(query)

    def _get_text_embedding(self, text: str) -> List[float]:
        """Get text embedding."""
        return self._embedding_model.embed_documents([text])[0]

    async def _aget_text_embedding(self, text: str) -> List[float]:
        """Get text embedding (async)."""
        try:
            embeds = await self._embedding_model.aembed_documents([text])
            return embeds[0]
        except NotImplementedError:
            return self._get_text_embedding(text)

    def _get_text_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Get text embeddings in batch."""
        return self._embedding_model.embed_documents(texts)
