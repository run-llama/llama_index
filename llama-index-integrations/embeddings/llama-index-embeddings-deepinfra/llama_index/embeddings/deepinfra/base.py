"""Embedding adapter model."""

import logging
from typing import Any, List, Optional, Type, cast

from llama_index.core.base.embeddings.base import BaseEmbedding
from llama_index.core.bridge.pydantic import PrivateAttr
from llama_index.core.callbacks import CallbackManager
from llama_index.core.constants import DEFAULT_EMBED_BATCH_SIZE
from llama_index.core.utils import infer_torch_device

logger = logging.getLogger(__name__)

DEFAULT_MODEL_ID = "sentence-transformers/clip-ViT-B-32"
MAX_BATCH_SIZE = 1024


class DeepInfraEmbeddingModel(BaseEmbedding):
    """DeepInfra embedding model

    API token can be passed as an argument or
    DEEPINFRA_API_TOKEN should be set in the environment variables.

    This is a wrapper around any embedding model that is available in DeepInfra.
    Check DeepInfra's website for the list of available models.
    https://deepinfra.com/models/embeddings

    Args:
        model_id (str): Model ID. Defaults to DEFAULT_MODEL_ID.
        normalize (bool): Whether to normalize embeddings. Defaults to False.

    Examples:
        >>> from llama_index.embeddings import DeepInfraEmbeddingModel
        >>> model = DeepInfraEmbeddingModel()
        >>> model.get_text_embedding("Hello, world!")
        [0.1, 0.2, 0.3, ...]


    """

    _model_id : string = PrivateAttr()
    _normalize : bool = PrivateAttr()
    _api_token : string = PrivateAttr()

    def __init__(
        self,
        model_id: string = DEFAULT_MODEL_ID,
        normalize: bool = False,
        api_token : string = None
    ) -> None:
        """Init params."""
        self._model_id = model_id
        self._normalize = normalize
        self._api_token = api_token or get_from_param_or_env("DEEPINFRA_API_TOKEN")


    def _get_query_embedding(self, query: str) -> List[float]:
        """Get query embedding."""
        return self._get_query_embedding(query)

    async def _aget_query_embedding(self, query: str) -> List[float]:
        """Get query embedding."""
        raise NotImplementedError

    def _get_text_embedding(self, text: str) -> List[float]:
        """Get text embedding."""
        raise NotImplementedError

    async def _aget_text_embedding(self, text: str) -> List[float]:
        """Async get text embedding."""
        raise NotImplementedError


    def _get_text_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Get text embedding."""
        return self._get_text_embedding(text)

    async def _aget_text_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Async get text embeddings."""
        return self._aget_text_embedding(text)




