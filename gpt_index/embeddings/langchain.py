"""Langchain Embedding Wrapper Module."""


from typing import List

from langchain.embeddings.base import Embeddings as LCEmbeddings

from gpt_index.embeddings.base import BaseEmbedding


class LangchainEmbedding(BaseEmbedding):
    """External embeddings (taken from Langchain).

    Args:
        langchain_embedding (langchain.embeddings.Embeddings): Langchain
            embeddings class.
    """

    def __init__(self, langchain_embedding: LCEmbeddings) -> None:
        """Init params."""
        super().__init__()
        self._langchain_embedding = langchain_embedding

    def _get_query_embedding(self, query: str) -> List[float]:
        """Get query embedding."""
        return self._langchain_embedding.embed_query(query)

    def _get_text_embedding(self, text: str) -> List[float]:
        """Get text embedding."""
        return self._langchain_embedding.embed_documents([text])[0]
