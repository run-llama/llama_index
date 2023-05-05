"""Test milvus index."""


from typing import Any, List, Optional
from llama_index.indices.service_context import ServiceContext
from llama_index.indices.vector_store import GPTVectorStoreIndex
from llama_index.storage.storage_context import StorageContext

from llama_index.vector_stores.types import (
    NodeWithEmbedding,
    VectorStore,
    VectorStoreQuery,
    VectorStoreQueryResult,
)


class MockMilvusVectorStore(VectorStore):
    stores_text: bool = True

    def __init__(
        self,
        collection_name: str = "llamalection",
        index_params: Optional[dict] = None,
        search_params: Optional[dict] = None,
        dim: Optional[int] = None,
        host: str = "localhost",
        port: int = 19530,
        user: str = "",
        password: str = "",
        use_secure: bool = False,
        overwrite: bool = False,
        **kwargs: Any,
    ) -> None:
        self.collection_name = collection_name
        self.index_params = index_params
        self.search_params = search_params
        self.dim = dim
        self.host = host
        self.port = port
        self.user = user
        self.password = password
        self.use_secure = use_secure
        self.overwrite = overwrite

    @property
    def client(self) -> Any:
        return None

    def add(
        self,
        embedding_results: List[NodeWithEmbedding],
    ) -> List[str]:
        return []

    def delete(self, doc_id: str, **delete_kwargs: Any) -> None:
        return None

    def query(self, query: VectorStoreQuery) -> VectorStoreQueryResult:
        return VectorStoreQueryResult()


def test_basic(mock_service_context: ServiceContext) -> None:
    """Test we can save and load."""
    vector_store = MockMilvusVectorStore()
    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    GPTVectorStoreIndex.from_documents(
        documents=[],
        storage_context=storage_context,
        service_context=mock_service_context,
    )
