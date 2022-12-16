"""OpenAI embeddings file."""

from enum import Enum
from typing import List

from openai.embeddings_utils import cosine_similarity, get_embedding

from gpt_index.embeddings.base import EMB_TYPE, BaseEmbedding


class OpenAIEmbeddingMode(str, Enum):
    """OpenAI embedding mode."""

    SIMILARITY_MODE = "similarity"
    TEXT_SEARCH_MODE = "text_search"


# convenient shorthand
OAEM = OpenAIEmbeddingMode


EMBED_MAX_TOKEN_LIMIT = 2048

# TODO: make enum
TEXT_SIMILARITY_DAVINCI = "text-similarity-davinci-001"
TEXT_SEARCH_DAVINCI_QUERY = "text-search-davinci-query-001"
TEXT_SEARCH_DAVINCI_DOC = "text-search-davinci-doc-001"

TEXT_SIMILARITY_CURIE = "text-similarity-curie-001"
TEXT_SEARCH_CURIE_QUERY = "text-search-curie-query-001"
TEXT_SEARCH_CURIE_DOC = "text-search-curie-doc-001"

TEXT_SIMILARITY_BABBAGE = "text-similarity-babbage-001"
TEXT_SEARCH_BABBAGE_QUERY = "text-search-babbage-query-001"
TEXT_SEARCH_BABBAGE_DOC = "text-search-babbage-doc-001"

TEXT_SIMILARITY_ADA = "text-similarity-ada-001"
TEXT_SEARCH_ADA_QUERY = "text-search-ada-query-001"
TEXT_SEARCH_ADA_DOC = "text-search-ada-doc-001"

# embedding-ada-002
TEXT_EMBED_ADA_002 = "text-embedding-ada-002"


_QUERY_MODE_MODEL_DICT = {
    (OAEM.SIMILARITY_MODE, "davinci"): TEXT_SIMILARITY_DAVINCI,
    (OAEM.SIMILARITY_MODE, "curie"): TEXT_SIMILARITY_CURIE,
    (OAEM.SIMILARITY_MODE, "babbage"): TEXT_SIMILARITY_BABBAGE,
    (OAEM.SIMILARITY_MODE, "ada"): TEXT_SIMILARITY_ADA,
    (OAEM.SIMILARITY_MODE, "text-embedding-ada-002"): TEXT_EMBED_ADA_002,
    (OAEM.TEXT_SEARCH_MODE, "davinci"): TEXT_SEARCH_DAVINCI_QUERY,
    (OAEM.TEXT_SEARCH_MODE, "curie"): TEXT_SEARCH_CURIE_QUERY,
    (OAEM.TEXT_SEARCH_MODE, "babbage"): TEXT_SEARCH_BABBAGE_QUERY,
    (OAEM.TEXT_SEARCH_MODE, "ada"): TEXT_SEARCH_ADA_QUERY,
    (OAEM.TEXT_SEARCH_MODE, "text-embedding-ada-002"): TEXT_EMBED_ADA_002,
}

_TEXT_MODE_MODEL_DICT = {
    (OAEM.SIMILARITY_MODE, "davinci"): TEXT_SIMILARITY_DAVINCI,
    (OAEM.SIMILARITY_MODE, "curie"): TEXT_SIMILARITY_CURIE,
    (OAEM.SIMILARITY_MODE, "babbage"): TEXT_SIMILARITY_BABBAGE,
    (OAEM.SIMILARITY_MODE, "ada"): TEXT_SIMILARITY_ADA,
    (OAEM.SIMILARITY_MODE, "text-embedding-ada-002"): TEXT_EMBED_ADA_002,
    (OAEM.TEXT_SEARCH_MODE, "davinci"): TEXT_SEARCH_DAVINCI_DOC,
    (OAEM.TEXT_SEARCH_MODE, "curie"): TEXT_SEARCH_CURIE_DOC,
    (OAEM.TEXT_SEARCH_MODE, "babbage"): TEXT_SEARCH_BABBAGE_DOC,
    (OAEM.TEXT_SEARCH_MODE, "ada"): TEXT_SEARCH_ADA_DOC,
    (OAEM.TEXT_SEARCH_MODE, "text-embedding-ada-002"): TEXT_EMBED_ADA_002,
}


class OpenAIEmbedding(BaseEmbedding):
    """OpenAI class for embeddings."""

    def __init__(
        self,
        mode: str = OpenAIEmbeddingMode.TEXT_SEARCH_MODE,
        model: str = "text-embedding-ada-002",
    ) -> None:
        """Init params."""
        self.mode = OpenAIEmbeddingMode(mode)
        self.model = model

    def get_query_embedding(self, query: str) -> List[float]:
        """Get query embedding."""
        key = (self.mode, self.model)
        if key not in _QUERY_MODE_MODEL_DICT:
            raise ValueError(f"Invalid mode, model combination: {key}")
        engine = _QUERY_MODE_MODEL_DICT[key]
        return get_embedding(query, engine=engine)

    def get_text_embedding(self, text: str) -> List[float]:
        """Get text embedding."""
        key = (self.mode, self.model)
        if key not in _TEXT_MODE_MODEL_DICT:
            raise ValueError(f"Invalid mode, model combination: {key}")
        engine = _TEXT_MODE_MODEL_DICT[key]
        return get_embedding(text, engine=engine)

    def similarity(self, embedding1: EMB_TYPE, embedding2: EMB_TYPE) -> float:
        """Get embedding similarity."""
        return cosine_similarity(embedding1, embedding2)
