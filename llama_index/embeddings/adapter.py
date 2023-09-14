"""Embedding adapter model."""

from typing import Any, List, Optional

from llama_index.bridge.pydantic import PrivateAttr

from llama_index.callbacks import CallbackManager
from llama_index.embeddings.base import DEFAULT_EMBED_BATCH_SIZE, BaseEmbedding

import logging

logger = logging.getLogger(__name__)


class LinearAdapterEmbeddingModel(BaseEmbedding):
    """Linear adapter for any embedding model."""

    _base_embed_model: BaseEmbedding = PrivateAttr()
    _adapter_path: str = PrivateAttr()
    _adapter: Any = PrivateAttr()
    _transform_query: bool = PrivateAttr()
    _device: Optional[str] = PrivateAttr()
    _target_device: Any = PrivateAttr()

    def __init__(
        self,
        base_embed_model: BaseEmbedding,
        adapter_path: str,
        transform_query: bool = True,
        device: Optional[str] = None,
        embed_batch_size: int = DEFAULT_EMBED_BATCH_SIZE,
        callback_manager: Optional[CallbackManager] = None,
    ) -> None:
        """Init params"""
        import torch
        from llama_index.embeddings.adapter_utils import LinearLayer

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
            logger.info("Use pytorch device: {}".format(device))
        self._target_device = torch.device(device)

        self._base_embed_model = base_embed_model
        self._adapter_path = adapter_path

        adapter = LinearLayer.load(adapter_path)
        self._adapter = adapter
        self._adapter.to(self._target_device)

        self._transform_query = transform_query
        super().__init__(
            embed_batch_size=embed_batch_size,
            callback_manager=callback_manager,
            model_name=f"Adapter for {base_embed_model.model_name}",
        )

    @classmethod
    def class_name(cls) -> str:
        """Get class name."""
        return "LinearAdapterEmbeddingModel"

    def _get_query_embedding(self, query: str) -> List[float]:
        """Get query embedding."""
        import torch

        query_embedding = self._base_embed_model._get_query_embedding(query)
        if self._transform_query:
            query_embedding_t = torch.tensor(query_embedding).to(self._target_device)
            query_embedding_t = self._adapter.forward(query_embedding_t)
            query_embedding = query_embedding_t.tolist()

        return query_embedding

    async def _aget_query_embedding(self, query: str) -> List[float]:
        """Get query embedding."""
        import torch

        query_embedding = await self._base_embed_model._aget_query_embedding(query)
        if self._transform_query:
            query_embedding_t = torch.tensor(query_embedding).to(self._target_device)
            query_embedding_t = self._adapter.forward(query_embedding_t)
            query_embedding = query_embedding_t.tolist()

        return query_embedding

    def _get_text_embedding(self, text: str) -> List[float]:
        text_embedding = self._base_embed_model._get_text_embedding(text)

        return text_embedding

    async def _aget_text_embedding(self, text: str) -> List[float]:
        text_embedding = await self._base_embed_model._aget_text_embedding(text)

        return text_embedding
