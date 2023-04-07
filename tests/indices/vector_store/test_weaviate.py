from typing import Any, Dict, List, Optional
from unittest.mock import Mock, patch

from gpt_index.indices.vector_store.vector_indices import GPTWeaviateIndex
from gpt_index.vector_stores.types import NodeEmbeddingResult
from tests.mock_utils.mock_decorator import patch_common


class MockWeaviateVectorStore:
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


@patch_common
@patch(
    "gpt_index.indices.vector_store.vector_indices.WeaviateVectorStore",
    MockWeaviateVectorStore,
)
@patch(
    "gpt_index.vector_stores.registry.VECTOR_STORE_CLASS_TO_VECTOR_STORE_TYPE",
    {MockWeaviateVectorStore: "mock_type"},
)
@patch(
    "gpt_index.vector_stores.registry.VECTOR_STORE_TYPE_TO_VECTOR_STORE_CLASS",
    {"mock_type": MockWeaviateVectorStore},
)
def test_save_load(
    _mock_init: Any,
    _mock_predict: Any,
    _mock_total_tokens_used: Any,
    _mock_split_text_overlap: Any,
    _mock_split_text: Any,
) -> None:
    """Test we can save and load."""
    weaviate_client = Mock()
    index = GPTWeaviateIndex.from_documents(
        documents=[], weaviate_client=weaviate_client
    )
    save_dict = index.save_to_dict()
    loaded_index = GPTWeaviateIndex.load_from_dict(
        save_dict,
        weaviate_client=weaviate_client,
    )
    assert isinstance(loaded_index, GPTWeaviateIndex)
