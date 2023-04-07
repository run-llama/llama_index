from typing import Any, Dict, List, Optional

from gpt_index.constants import DATA_KEY, TYPE_KEY
from gpt_index.indices.composability.utils import (
    load_query_context_from_dict,
    save_query_context_to_dict,
)
from gpt_index.vector_stores.types import (
    NodeEmbeddingResult,
    VectorStore,
    VectorStoreQueryResult,
    VectorStoreQuery,
)


class MockVectorStore(VectorStore):
    stores_text: bool = True

    def __init__(self, config_dict: Optional[Dict[str, Any]] = None) -> None:
        self._config_dict = config_dict or {
            "attr1": 0,
            "attr2": "attr2_val",
        }

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "MockVectorStore":
        return cls(config_dict)

    @property
    def config_dict(self) -> Dict[str, Any]:
        return self._config_dict

    @property
    def client(self) -> Any:
        """Get client."""
        return None

    def add(
        self,
        embedding_results: List[NodeEmbeddingResult],
    ) -> List[str]:
        """Add embedding results to vector store."""
        raise NotImplementedError()

    def delete(self, doc_id: str, **delete_kwargs: Any) -> None:
        """Delete doc."""
        raise NotImplementedError()

    def query(self, query: VectorStoreQuery) -> VectorStoreQueryResult:
        """Query vector store."""
        raise NotImplementedError()


def test_save_query_context_to_dict() -> None:
    """Test save query context to dict."""
    vector_store = MockVectorStore()
    query_context = {"test_index_id": {"vector_store": vector_store}}

    expected_dict = {
        "test_index_id": {
            "vector_store": {
                TYPE_KEY: "mock_type",
                DATA_KEY: vector_store.config_dict,
            }
        }
    }

    save_dict = save_query_context_to_dict(
        query_context,
        vector_store_cls_to_type={MockVectorStore: "mock_type"},  # type:ignore
    )

    assert save_dict == expected_dict


def test_load_query_context_from_dict() -> None:
    """Test load query context from dict."""
    vector_store = MockVectorStore()

    save_dict = {
        "test_index_id": {
            "vector_store": {
                TYPE_KEY: "mock_type",
                DATA_KEY: vector_store.config_dict,
            }
        }
    }

    # Test without kwargs
    query_context = load_query_context_from_dict(
        save_dict,
        vector_store_type_to_cls={"mock_type": MockVectorStore},  # type:ignore
    )
    loaded_vector_store = query_context["test_index_id"]["vector_store"]
    assert isinstance(loaded_vector_store, MockVectorStore)

    # Test with kwargs
    query_context_kwargs = {
        "test_index_id": {"vector_store": {"extra_key": "extra_value"}}
    }
    query_context = load_query_context_from_dict(
        save_dict,
        vector_store_type_to_cls={"mock_type": MockVectorStore},  # type:ignore
        query_context_kwargs=query_context_kwargs,
    )
    loaded_vector_store = query_context["test_index_id"]["vector_store"]
    assert isinstance(loaded_vector_store, MockVectorStore)
    assert loaded_vector_store.config_dict["extra_key"] == "extra_value"
