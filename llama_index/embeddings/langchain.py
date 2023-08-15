"""Langchain Embedding Wrapper Module."""

from typing import List, Optional

from llama_index.bridge.langchain import Embeddings as LCEmbeddings
from llama_index.callbacks import CallbackManager
from llama_index.embeddings.base import BaseEmbedding, DEFAULT_EMBED_BATCH_SIZE


class LangchainEmbedding(BaseEmbedding):
    """External embeddings (taken from Langchain).

    Args:
        langchain_embedding (langchain.embeddings.Embeddings): Langchain
            embeddings class.
    """

    _langchain_embedding: LCEmbeddings

    def __init__(
        self,
        langchain_embeddings: LCEmbeddings,
        model_name: Optional[str] = None,
        embed_batch_size: int = DEFAULT_EMBED_BATCH_SIZE,
        callback_manager: Optional[CallbackManager] = None,
    ):
        # attempt to get a useful model name
        if model_name is not None:
            model_name = model_name
        elif hasattr(langchain_embeddings, "model_name"):
            model_name = langchain_embeddings.model_name
        elif hasattr(langchain_embeddings, "model"):
            model_name = langchain_embeddings.model
        else:
            model_name = type(langchain_embeddings).__name__

        super().__init__(
            langchain_embeddings=langchain_embeddings,
            embed_batch_size=embed_batch_size,
            callback_manager=callback_manager,
            model_name=model_name,
        )

    def _get_query_embedding(self, query: str) -> List[float]:
        """Get query embedding."""
        return self._langchain_embedding.embed_query(query)

    async def _aget_query_embedding(self, query: str) -> List[float]:
        return await self._langchain_embedding.aembed_query(query)

    async def _aget_text_embedding(self, text: str) -> List[float]:
        embeds = await self._langchain_embedding.aembed_documents([text])
        return embeds[0]

    def _get_text_embedding(self, text: str) -> List[float]:
        """Get text embedding."""
        return self._langchain_embedding.embed_documents([text])[0]

    def _get_text_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Get text embeddings."""
        return self._langchain_embedding.embed_documents(texts)
