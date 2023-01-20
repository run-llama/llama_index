"""OpenAI embeddings file."""

from enum import Enum
from typing import List, Optional

import openai
from tenacity import retry, stop_after_attempt, wait_random_exponential

from gpt_index.embeddings.base import BaseEmbedding


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


@retry(wait=wait_random_exponential(min=1, max=20), stop=stop_after_attempt(6))
def get_embedding(
    text: str,
    engine: Optional[str] = None,
) -> List[float]:
    """Get embedding.

    NOTE: Copied from OpenAI's embedding utils:
    https://github.com/openai/openai-python/blob/main/openai/embeddings_utils.py

    Copied here to avoid importing unnecessary dependencies
    like matplotlib, plotly, scipy, sklearn.

    """
    text = text.replace("\n", " ")
    return openai.Embedding.create(input=[text], engine=engine)["data"][0]["embedding"]


class OpenAIEmbedding(BaseEmbedding):
    """OpenAI class for embeddings.

    Args:
        mode (str): Mode for embedding.
            Defaults to OpenAIEmbeddingMode.TEXT_SEARCH_MODE.
            Options are:

            - OpenAIEmbeddingMode.SIMILARITY_MODE
            - OpenAIEmbeddingMode.TEXT_SEARCH_MODE

        model (str): Model for embedding. Defaults to "text-embedding-ada-002".
            Options are:

            - "davinci"
            - "curie"
            - "babbage"
            - "ada"
            - "text-embedding-ada-002"

        deployment_name (Optional[str]): Optional deployment of model. Defaults to None.
            If this value is not None, mode and model will be ignored.
            Only available for using AzureOpenAI.
    """

    def __init__(
        self,
        mode: str = OpenAIEmbeddingMode.TEXT_SEARCH_MODE,
        model: str = "text-embedding-ada-002",
        deployment_name: Optional[str] = None,
    ) -> None:
        """Init params."""
        super().__init__()
        self.mode = OpenAIEmbeddingMode(mode)
        self.model = model
        self.deployment_name = deployment_name

    def _get_query_embedding(self, query: str) -> List[float]:
        """Get query embedding."""
        if self.deployment_name is not None:
            engine = self.deployment_name
        else:
            key = (self.mode, self.model)
            if key not in _QUERY_MODE_MODEL_DICT:
                raise ValueError(f"Invalid mode, model combination: {key}")
            engine = _QUERY_MODE_MODEL_DICT[key]
        return get_embedding(query, engine=engine)

    def _get_text_embedding(self, text: str) -> List[float]:
        """Get text embedding."""
        if self.deployment_name is not None:
            engine = self.deployment_name
        else:
            key = (self.mode, self.model)
            if key not in _TEXT_MODE_MODEL_DICT:
                raise ValueError(f"Invalid mode, model combination: {key}")
            engine = _TEXT_MODE_MODEL_DICT[key]
        return get_embedding(text, engine=engine)
