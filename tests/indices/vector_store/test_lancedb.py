"""Test LanceDB index."""
from typing import Any, List, Optional, Dict

from gpt_index.indices.vector_store import GPTVectorStoreIndex
from gpt_index.indices.service_context import ServiceContext
from gpt_index.storage.storage_context import StorageContext

from gpt_index.vector_stores.types import (
    NodeEmbeddingResult,
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

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "MockLanceDBVectorStore":
        return cls(**config_dict)

    @property
    def client(self) -> None:
        """Get client."""
        return None

    @property
    def config_dict(self) -> dict:
        """Return config dict."""
        return {
            "uri": self.uri,
            "table_name": self.table_name,
            "nprobes": self.nprobes,
            "refine_factor": self.refine_factor,
        }

    def add(
        self,
        embedding_results: List[NodeEmbeddingResult],
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
