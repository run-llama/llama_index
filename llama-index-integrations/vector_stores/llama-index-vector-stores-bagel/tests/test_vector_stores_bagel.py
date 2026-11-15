import math
from unittest.mock import MagicMock

import pytest
from llama_index.core.vector_stores.types import (
    BasePydanticVectorStore,
    VectorStoreQuery,
)
from llama_index.vector_stores.bagel import BagelVectorStore


def test_class():
    names_of_base_classes = [b.__name__ for b in BagelVectorStore.__mro__]
    assert BasePydanticVectorStore.__name__ in names_of_base_classes


def test_query_similarity_decreases_with_distance() -> None:
    collection = MagicMock()
    collection.find.return_value = {
        "ids": [["closest", "farther"]],
        "documents": [["closest text", "farther text"]],
        "metadatas": [[{}, {}]],
        "distances": [[0.0, 1.0]],
    }
    # model_construct skips the bagel Cluster type check in __init__
    store = BagelVectorStore.model_construct()
    store._collection = collection

    result = store.query(
        VectorStoreQuery(query_embedding=[0.1, 0.2], similarity_top_k=2)
    )

    assert result.ids == ["closest", "farther"]
    assert result.similarities == pytest.approx([1.0, math.exp(-1.0)])
