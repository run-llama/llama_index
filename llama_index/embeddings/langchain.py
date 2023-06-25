"""Langchain Embedding Wrapper Module."""


from typing import Any, List

from llama_index.bridge.langchain import Embeddings as LCEmbeddings

from llama_index.embeddings.base import BaseEmbedding


class LangchainEmbedding(BaseEmbedding):
    """External embeddings (taken from Langchain).

    Args:
        langchain_embedding (langchain.embeddings.Embeddings): Langchain
            embeddings class.
    """

    def __init__(self, langchain_embedding: LCEmbeddings, **kwargs: Any) -> None:
        """Init params."""
        super().__init__(**kwargs)
        self._langchain_embedding = langchain_embedding

    def _get_query_embedding(self, query: str) -> List[float]:
        """Get query embedding."""
        return self._langchain_embedding.embed_query(query)

    def _get_text_embedding(self, text: str) -> List[float]:
        """Get text embedding."""
        return self._langchain_embedding.embed_documents([text])[0]

    def _get_text_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Get text embeddings."""
        return self._langchain_embedding.embed_documents(texts)
