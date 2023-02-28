"""Base vector store index query."""


from typing import Any, List, Optional

from gpt_index.data_structs.data_structs import IndexDict
from gpt_index.indices.query.base import BaseGPTIndexQuery
from gpt_index.indices.query.vector_store.base import GPTVectorStoreIndexQuery


class GPTSimpleVectorIndexQuery(GPTVectorStoreIndexQuery[IndexDict]):
    """GPT simple vector index query.

    Args:
        embed_model (Optional[BaseEmbedding]): embedding model
        similarity_top_k (int): number of top k results to return
        vector_store (Optional[VectorStore]): vector store

    """

    def __init__(
        self,
        index_struct: IndexDict,
        **kwargs: Any,
    ) -> None:
        """Initialize params."""
        super().__init__(index_struct=index_struct, embed_model=embed_model, **kwargs)
