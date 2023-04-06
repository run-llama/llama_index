from typing import Any, Dict

from gpt_index.constants import DATA_KEY, TYPE_KEY
from gpt_index.indices.composability.utils import (
    load_query_context_from_dict,
    save_query_context_to_dict,
)
from gpt_index.vector_stores.types import VectorStore


class MockVectorStore(VectorStore):
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "MockVectorStore":
        del config_dict
        return cls()

    @property
    def config_dict(self) -> Dict[str, Any]:
        return {
            "attr1": 0,
            "attr2": "attr2_val",
        }


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
        query_context, vector_store_cls_to_type={MockVectorStore: "mock_type"}
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

    query_context = load_query_context_from_dict(
        save_dict, vector_store_type_to_cls={"mock_type": MockVectorStore}
    )
    loaded_vector_store = query_context["test_index_id"]["vector_store"]
    assert isinstance(loaded_vector_store, MockVectorStore)
