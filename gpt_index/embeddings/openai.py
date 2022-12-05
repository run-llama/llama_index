"""OpenAI embeddings file."""

from enum import Enum
from typing import List

from openai.embeddings_utils import cosine_similarity, get_embedding

from gpt_index.embeddings.base import EMB_TYPE, BaseEmbedding


class OpenAIEmbeddingMode(str, Enum):
    """OpenAI embedding mode."""

    SIMILARITY_MODE = "similarity"
    TEXT_SEARCH_MODE = "text_search"


TEXT_SIMILARITY_DAVINCI = "text-similarity-davinci-001"
TEXT_SEARCH_DAVINCI_QUERY = "text-search-davinci-query-001"
TEXT_SEARCH_DAVINCI_DOC = "text-search-davinci-doc-001"


class OpenAIEmbedding(BaseEmbedding):
    """OpenAI class for embeddings."""

    def __init__(self, mode: str = OpenAIEmbeddingMode.TEXT_SEARCH_MODE):
        """Init params."""
        self.mode = OpenAIEmbeddingMode(mode)

    def get_query_embedding(self, query: str) -> List[float]:
        """Get query embedding."""
        if self.mode == OpenAIEmbeddingMode.SIMILARITY_MODE:
            engine = TEXT_SIMILARITY_DAVINCI
        elif self.mode == OpenAIEmbeddingMode.TEXT_SEARCH_MODE:
            engine = TEXT_SEARCH_DAVINCI_QUERY
        return get_embedding(query, engine=engine)

    def get_text_embedding(self, text: str) -> List[float]:
        """Get text embedding."""
        if self.mode == OpenAIEmbeddingMode.SIMILARITY_MODE:
            engine = TEXT_SIMILARITY_DAVINCI
        elif self.mode == OpenAIEmbeddingMode.TEXT_SEARCH_MODE:
            engine = TEXT_SEARCH_DAVINCI_DOC
        return get_embedding(text, engine=engine)

    def similarity(self, embedding1: EMB_TYPE, embedding2: EMB_TYPE) -> float:
        """Get embedding similarity."""
        return cosine_similarity(embedding1, embedding2)
