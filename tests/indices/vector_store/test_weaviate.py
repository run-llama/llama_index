from typing import Any, List, Optional
from unittest.mock import Mock
from llama_index.indices.service_context import ServiceContext

from llama_index.indices.vector_store import GPTVectorStoreIndex
from llama_index.storage.storage_context import StorageContext
from llama_index.vector_stores.types import (
    NodeWithEmbedding,
    VectorStore,
    VectorStoreQuery,
    VectorStoreQueryResult,
)


class MockWeaviateVectorStore(VectorStore):
    stores_text: bool = True

    def __init__(
        self,
        weaviate_client: Optional[Any] = None,
        class_prefix: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        self.weaviate_client = weaviate_client
        self._class_prefix = class_prefix

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
    weaviate_client = Mock()
    vector_store = MockWeaviateVectorStore(weaviate_client=weaviate_client)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    GPTVectorStoreIndex.from_documents(
        documents=[],
        storage_context=storage_context,
        service_context=mock_service_context,
    )
