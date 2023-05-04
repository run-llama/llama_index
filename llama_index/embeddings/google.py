"""Google Universal Sentence Encoder Embedding Wrapper Module."""

from typing import List, Optional


from llama_index.embeddings.base import BaseEmbedding

# Google Universal Sentence Encode v5
DEFAULT_HANDLE = "https://tfhub.dev/google/universal-sentence-encoder-large/5"


class GoogleUnivSentEncoderEmbedding(BaseEmbedding):
    def __init__(self, handle: Optional[str] = None) -> None:
        """Init params."""
        handle = handle or DEFAULT_HANDLE
        try:
            import tensorflow_hub as hub

            self._google_use = hub.load(handle)
        except ImportError:
            raise ImportError(
                "Please install tensorflow_hub: `pip install tensorflow_hub`"
            )

        super().__init__()

    def _get_query_embedding(self, query: str) -> List[float]:
        """Get query embedding."""
        return self._get_embedding(query)

    def _get_text_embedding(self, text: str) -> List[float]:
        """Get text embedding."""
        return self._get_embedding(text)

    def _get_embedding(self, text: str) -> List[float]:
        vectors = self._google_use([text]).numpy().tolist()
        return vectors[0]
