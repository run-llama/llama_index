"""Base vector store index query."""


from abc import abstractmethod
from typing import Any, Generic, List, Optional, Tuple, TypeVar

from gpt_index.data_structs.data_structs import BaseIndexDict, Node
from gpt_index.embeddings.base import BaseEmbedding
from gpt_index.embeddings.openai import OpenAIEmbedding
from gpt_index.indices.query.base import BaseGPTIndexQuery
from gpt_index.indices.response.builder import ResponseBuilder, ResponseSourceBuilder
from gpt_index.indices.response.schema import Response
from gpt_index.indices.utils import truncate_text

BID = TypeVar("BID", bound=BaseIndexDict)


class BaseGPTVectorStoreIndexQuery(BaseGPTIndexQuery[BID], Generic[BID]):
    """Base vector store query."""

    def __init__(
        self,
        index_struct: BID,
        embed_model: Optional[BaseEmbedding] = None,
        similarity_top_k: Optional[int] = 1,
        **kwargs: Any,
    ) -> None:
        """Initialize params."""
        super().__init__(index_struct=index_struct, **kwargs)
        self._embed_model = embed_model or OpenAIEmbedding()
        self.similarity_top_k = similarity_top_k
