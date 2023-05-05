"""Test LanceDB index."""
from typing import Any, List, Optional

from llama_index.indices.vector_store import GPTVectorStoreIndex
from llama_index.indices.service_context import ServiceContext
from llama_index.storage.storage_context import StorageContext

from llama_index.vector_stores.types import (
    NodeWithEmbedding,
    VectorStore,
    VectorStoreQuery,
    VectorStoreQueryResult,
)


class MockLanceDBVectorStore(VectorStore):
    stores_text: bool = True

    def __init__(
        self,
        uri: str,
        table_name: str = "vectors",
        nprobes: int = 20,
        refine_factor: Optional[int] = None,
        **kwargs: Any,
    ) -> None:
        self.connection = None
        self.uri = uri
        self.table_name = table_name
        self.nprobes = nprobes
        self.refine_factor = refine_factor

    @property
    def client(self) -> None:
        """Get client."""
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


def test_save_load(mock_service_context: ServiceContext) -> None:
    """Test we can save and load."""
    vector_store = MockLanceDBVectorStore(uri=".")
    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    GPTVectorStoreIndex.from_documents(
        documents=[],
        service_context=mock_service_context,
        storage_context=storage_context,
    )
