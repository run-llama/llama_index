"""Test milvus index."""


from typing import Any, Dict, List, Optional
from unittest.mock import patch
from gpt_index.indices.vector_store.vector_indices import GPTMilvusIndex

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


@patch_common
@patch(
    "gpt_index.indices.vector_store.vector_indices.MilvusVectorStore",
    MockMilvusVectorStore,
)
@patch(
    "gpt_index.vector_stores.registry.VECTOR_STORE_CLASS_TO_VECTOR_STORE_TYPE",
    {MockMilvusVectorStore: "mock_type"},
)
@patch(
    "gpt_index.vector_stores.registry.VECTOR_STORE_TYPE_TO_VECTOR_STORE_CLASS",
    {"mock_type": MockMilvusVectorStore},
)
def test_save_load(
    _mock_init: Any,
    _mock_predict: Any,
    _mock_total_tokens_used: Any,
    _mock_split_text_overlap: Any,
    _mock_split_text: Any,
) -> None:
    """Test we can save and load."""
    index = GPTMilvusIndex.from_documents(documents=[])
    save_dict = index.save_to_dict()
    loaded_index = GPTMilvusIndex.load_from_dict(
        save_dict,
    )
    assert isinstance(loaded_index, GPTMilvusIndex)
