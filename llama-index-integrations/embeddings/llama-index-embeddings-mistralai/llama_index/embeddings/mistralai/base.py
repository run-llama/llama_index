"""MistralAI embeddings file."""

from typing import Any, List, Optional, Sequence, cast

from llama_index.core.base.embeddings.base import (
    DEFAULT_EMBED_BATCH_SIZE,
    BaseEmbedding,
)
from llama_index.core.bridge.pydantic import PrivateAttr
from llama_index.core.callbacks.base import CallbackManager
from llama_index.core.base.llms.generic_utils import get_from_param_or_env

try:
    # mistralai v1.x (<2): top-level client
    from mistralai import Mistral as MistralClientType
except ImportError:
    try:
        # mistralai v2.x (>=2): client under mistralai.client
        from mistralai.client import Mistral as MistralClientType
    except ImportError:
        # Older pre-v1 layouts
        from mistralai.client import MistralClient as MistralClientType


class MistralAIEmbedding(BaseEmbedding):
    """
    Class for MistralAI embeddings.

    Args:
        model_name (str): Model for embedding.
            Defaults to "mistral-embed".

        api_key (Optional[str]): API key to access the model. Defaults to None.

    """

    # Instance variables initialized via Pydantic's mechanism
    _client: Any = PrivateAttr()

    def __init__(
        self,
        model_name: str = "mistral-embed",
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
        api_key = get_from_param_or_env("api_key", api_key, "MISTRAL_API_KEY", "")

        if not api_key:
            raise ValueError(
                "You must provide an API key to use mistralai. "
                "You can either pass it in as an argument or set it `MISTRAL_API_KEY`."
            )
        self._client = MistralClientType(api_key=api_key)

    @classmethod
    def class_name(cls) -> str:
        return "MistralAIEmbedding"

    def _get_query_embedding(self, query: str) -> List[float]:
        """Get query embedding."""
        return self._create_embeddings([query])[0]

    async def _aget_query_embedding(self, query: str) -> List[float]:
        """The asynchronous version of _get_query_embedding."""
        return (await self._acreate_embeddings([query]))[0]

    def _get_text_embedding(self, text: str) -> List[float]:
        """Get text embedding."""
        return self._create_embeddings([text])[0]

    async def _aget_text_embedding(self, text: str) -> List[float]:
        """Asynchronously get text embedding."""
        return (await self._acreate_embeddings([text]))[0]

    def _get_text_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Get text embeddings."""
        return self._create_embeddings(texts)

    async def _aget_text_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Asynchronously get text embeddings."""
        return await self._acreate_embeddings(texts)

    def _create_embeddings(self, inputs: List[str]) -> List[List[float]]:
        embeddings_api = getattr(self._client, "embeddings", None)
        if embeddings_api is None:
            raise ValueError("Mistral client does not expose embeddings API.")

        if hasattr(embeddings_api, "create"):
            response = embeddings_api.create(model=self.model_name, inputs=inputs)
        else:
            # Legacy mistralai SDK shape: client.embeddings(model=..., input=...)
            response = embeddings_api(model=self.model_name, input=inputs)

        return self._parse_embedding_response(response)

    async def _acreate_embeddings(self, inputs: List[str]) -> List[List[float]]:
        embeddings_api = getattr(self._client, "embeddings", None)
        if embeddings_api is None:
            raise ValueError("Mistral client does not expose embeddings API.")

        if hasattr(embeddings_api, "create_async"):
            response = await embeddings_api.create_async(
                model=self.model_name, inputs=inputs
            )
            return self._parse_embedding_response(response)

        # Legacy SDK does not provide async embedding APIs.
        return self._create_embeddings(inputs)

    def _parse_embedding_response(self, response: Any) -> List[List[float]]:
        data = getattr(response, "data", None)
        if data is None and isinstance(response, dict):
            data = response.get("data", [])
        if data is None:
            raise ValueError("Unexpected Mistral embedding response format.")

        embeddings: List[List[float]] = []
        for item in cast(Sequence[Any], data):
            if isinstance(item, dict):
                embeddings.append(cast(List[float], item.get("embedding", [])))
            else:
                embeddings.append(cast(List[float], getattr(item, "embedding", [])))
        return embeddings
