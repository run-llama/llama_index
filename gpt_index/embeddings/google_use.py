"""Google Universal Sentence Encoder Embedding Wrapper Module."""

from typing import List

import tensorflow_hub as hub

from gpt_index.embeddings.base import BaseEmbedding

# Google Universal Sentence Encode v5
google_use = hub.load("https://tfhub.dev/google/universal-sentence-encoder-large/5")

def get_embedding(text: str) -> List[float]:
    vectors = google_use([text]).numpy().tolist()
    return vectors[0]
    
class GoogleUnivSentEncoderEmbedding(BaseEmbedding):

    def __init__(self) -> None:
        """Init params."""
        super().__init__()
        
    def _get_query_embedding(self, query: str) -> List[float]:
        """Get query embedding."""
        return get_embedding(query)

    def _get_text_embedding(self, text: str) -> List[float]:
        """Get text embedding."""
        return get_embedding(text)
