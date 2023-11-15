"""Langchain Embedding Wrapper Module."""

from typing import TYPE_CHECKING, List, Optional

from llama_index.bridge.pydantic import PrivateAttr
from llama_index.callbacks import CallbackManager
from llama_index.embeddings.base import DEFAULT_EMBED_BATCH_SIZE, BaseEmbedding

if TYPE_CHECKING:
    from llama_index.bridge.langchain import Embeddings as LCEmbeddings


class LangchainEmbedding(BaseEmbedding):
    """External embeddings (taken from Langchain).

    Args:
        langchain_embedding (langchain.embeddings.Embeddings): Langchain
            embeddings class.
    """

    _langchain_embedding: "LCEmbeddings" = PrivateAttr()
    _async_not_implemented_warned: bool = PrivateAttr(default=False)

    def __init__(
        self,
        langchain_embeddings: "LCEmbeddings",
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

        self._langchain_embedding = langchain_embeddings
        super().__init__(
            embed_batch_size=embed_batch_size,
            callback_manager=callback_manager,
            model_name=model_name,
        )

    @classmethod
    def class_name(cls) -> str:
        return "LangchainEmbedding"

    def _async_not_implemented_warn_once(self) -> None:
        if not self._async_not_implemented_warned:
            print("Async embedding not available, falling back to sync method.")
            self._async_not_implemented_warned = True

    def _get_query_embedding(self, query: str) -> List[float]:
        """Get query embedding."""
        return self._langchain_embedding.embed_query(query)

    async def _aget_query_embedding(self, query: str) -> List[float]:
        try:
            return await self._langchain_embedding.aembed_query(query)
        except NotImplementedError:
            # Warn the user that sync is being used
            self._async_not_implemented_warn_once()
            return self._get_query_embedding(query)

    async def _aget_text_embedding(self, text: str) -> List[float]:
        try:
            embeds = await self._langchain_embedding.aembed_documents([text])
            return embeds[0]
        except NotImplementedError:
            # Warn the user that sync is being used
            self._async_not_implemented_warn_once()
            return self._get_text_embedding(text)

    def _get_text_embedding(self, text: str) -> List[float]:
        """Get text embedding."""
        return self._langchain_embedding.embed_documents([text])[0]

    def _get_text_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Get text embeddings."""
        return self._langchain_embedding.embed_documents(texts)
