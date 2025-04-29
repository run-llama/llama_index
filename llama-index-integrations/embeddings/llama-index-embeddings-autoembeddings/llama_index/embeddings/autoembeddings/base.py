from typing import Any, List

from llama_index.core.base.embeddings.base import BaseEmbedding
from llama_index.core.bridge.pydantic import PrivateAttr, Field
from typing import Optional

try:
    import chonkie
    from chonkie import AutoEmbeddings
except ImportError:
    raise ImportError(
        "Could not import Autembeddings from chonkie. "
        "Please install it with `pip install chonkie[all]`."
    )


class ChonkieAutoEmbedding(BaseEmbedding):
    """
    """
    model_name: str 
    embedder: Optional[chonkie.BaseEmbeddings] = None
    
    def __init__(
        self, **kwargs: Any
    ) -> None:
        super().__init__(**kwargs)
        self.embedder = AutoEmbeddings.get_embeddings(self.model_name)
    @classmethod
    def class_name(cls) -> str:
        return "ChonkieAutoEmbedding"
    def _get_embedding(self, text: str) -> List[float]:
        embed = self.embedder.embed(text)
        return embed.tolist()
    async def _aget_embedding(self, text: str) -> List[float]:
        return self._get_embedding(text)
    def _get_embeddings(self, texts: List[str]) -> List[List[float]]:
        embeds = self.embedder.embed_batch(texts)
        embeds_list = [e.tolist() for e in embeds]
        return embeds_list
    async def _aget_embeddings(
        self,
        texts: List[str],
    ) -> List[List[float]]:
        return self._get_embeddings(texts)
    def _get_query_embedding(self, query: str) -> List[float]:
        """Get query embedding."""
        return self._get_embedding(query)
    async def _aget_query_embedding(self, query: str) -> List[float]:
        """Get query embedding."""
        return await self._aget_embedding(query)
    def _get_text_embedding(self, text: str) -> List[float]:
        """Get text embedding."""
        return self._get_embedding(text)