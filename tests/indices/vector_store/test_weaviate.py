from typing import Any, Dict, List, Optional
from unittest.mock import Mock
from gpt_index.indices.service_context import ServiceContext

from gpt_index.indices.vector_store import GPTVectorStoreIndex
from gpt_index.storage.storage_context import StorageContext
from gpt_index.vector_stores.types import NodeEmbeddingResult, VectorStore


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
    def config_dict(self) -> dict:
        return {"class_prefix": "test_class_prefix"}

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "MockWeaviateVectorStore":
        if "weaviate_client" not in config_dict:
            raise ValueError("Missing Weaviate client!")
        return cls(**config_dict)

    def add(
        self,
        embedding_results: List[NodeEmbeddingResult],
    ) -> List[str]:
        return []


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
