"""Google Universal Sentence Encoder Embedding Wrapper Module."""

from typing import Any, List, Optional

from llama_index.bridge.pydantic import PrivateAttr
from llama_index.callbacks import CallbackManager
from llama_index.embeddings.base import DEFAULT_EMBED_BATCH_SIZE, BaseEmbedding

# Google Universal Sentence Encode v5
DEFAULT_HANDLE = "https://tfhub.dev/google/universal-sentence-encoder-large/5"


class GoogleUnivSentEncoderEmbedding(BaseEmbedding):
    _model: Any = PrivateAttr()

    def __init__(
        self,
        handle: Optional[str] = None,
        embed_batch_size: int = DEFAULT_EMBED_BATCH_SIZE,
        callback_manager: Optional[CallbackManager] = None,
    ):
        """Init params."""
        handle = handle or DEFAULT_HANDLE
        try:
            import tensorflow_hub as hub

            model = hub.load(handle)
        except ImportError:
            raise ImportError(
                "Please install tensorflow_hub: `pip install tensorflow_hub`"
            )

        self._model = model
        super().__init__(
            embed_batch_size=embed_batch_size,
            callback_manager=callback_manager,
            model_name=handle,
        )

    @classmethod
    def class_name(cls) -> str:
        return "GoogleUnivSentEncoderEmbedding"

    def _get_query_embedding(self, query: str) -> List[float]:
        """Get query embedding."""
        return self._get_embedding(query)

    # TODO: use proper async methods
    async def _aget_text_embedding(self, query: str) -> List[float]:
        """Get text embedding."""
        return self._get_embedding(query)

    # TODO: user proper async methods
    async def _aget_query_embedding(self, query: str) -> List[float]:
        """Get query embedding."""
        return self._get_embedding(query)

    def _get_text_embedding(self, text: str) -> List[float]:
        """Get text embedding."""
        return self._get_embedding(text)

    def _get_embedding(self, text: str) -> List[float]:
        vectors = self._model([text]).numpy().tolist()
        return vectors[0]
