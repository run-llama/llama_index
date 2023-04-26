"""Test milvus index."""


from typing import Any, Dict, List, Optional
from unittest.mock import patch
from gpt_index.indices.service_context import ServiceContext
from gpt_index.indices.vector_store import GPTVectorStoreIndex
from gpt_index.storage.storage_context import StorageContext

from gpt_index.vector_stores.types import NodeEmbeddingResult
from tests.mock_utils.mock_decorator import patch_common


class MockMilvusVectorStore:
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
    def config_dict(self) -> dict:
        """Return config dict."""
        return {
            "collection_name": self.collection_name,
            "index_params": self.index_params,
            "search_params": self.search_params,
            "dim": self.dim,
            "host": self.host,
            "port": self.port,
            "user": self.user,
            "password": self.password,
            "use_secure": self.use_secure,
            # # Set to false, dont want subsequent object to rewrite store
            # "overwrite": False,
        }

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "MockMilvusVectorStore":
        return cls(**config_dict)

    def add(
        self,
        embedding_results: List[NodeEmbeddingResult],
    ) -> List[str]:
        return []


def test_basic(mock_service_context: ServiceContext) -> None:
    """Test we can save and load."""
    vector_store = MockMilvusVectorStore()
    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    GPTVectorStoreIndex.from_documents(
        documents=[],
        storage_context=storage_context,
        service_context=mock_service_context,
    )
