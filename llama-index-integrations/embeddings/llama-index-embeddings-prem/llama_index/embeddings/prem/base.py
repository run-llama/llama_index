"""PremAI embeddings file."""

from typing import Any, List, Optional

from llama_index.core.base.embeddings.base import (
    DEFAULT_EMBED_BATCH_SIZE,
    BaseEmbedding,
)
from llama_index.core.bridge.pydantic import PrivateAttr
from llama_index.core.callbacks.base import CallbackManager
from llama_index.core.base.llms.generic_utils import get_from_param_or_env

from premai import Prem


class PremAIEmbeddings(BaseEmbedding):
    """Class for PremAI embeddings.

    Args:
        model_name (str): Model for embedding.
            Defaults to "mistral-embed".

        api_key (Optional[str]): API key to access the model. Defaults to None.
    """

    # Instance variables initialized via Pydantic's mechanism
    _premai_client: Any = PrivateAttr()

    def __init__(
        self,
        project_id: int,
        model_name: str,
        api_key: Optional[str] = None,
        callback_manager: Optional[CallbackManager] = None,
        **kwargs: Any,
    ):
        api_key = get_from_param_or_env("api_key", api_key, "PREMAI_API_KEY", "")

        if not api_key:
            raise ValueError(
                "You must provide an API key to use PremAI. "
                "You can either pass it in as an argument or set it `PREMAI_API_KEY`."
            )
        self._mistralai_client = Prem(api_key=api_key)
        super().__init__(
            project_id=project_id,
            model_name=model_name,
            callback_manager=callback_manager,
            **kwargs,
        )

    @classmethod
    def class_name(cls) -> str:
        return "PremAIEmbedding"

    def _get_query_embedding(self, query: str) -> List[float]:
        """Get query embedding."""
        return (
            self._mistralai_client.embeddings(model=self.model_name, input=[query])
            .data[0]
            .embedding
        )

    async def _aget_query_embedding(self, query: str) -> List[float]:
        raise NotImplementedError("Async calls are not available in this version.")

    def _get_text_embedding(self, text: str) -> List[float]:
        """Get text embedding."""
        return (
            self._mistralai_client.embeddings(model=self.model_name, input=[text])
            .data[0]
            .embedding
        )

    def _get_text_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Get text embeddings."""
        embedding_response = self._mistralai_client.embeddings(
            model=self.model_name, input=texts
        ).data
        return [embed.embedding for embed in embedding_response]
