from unittest.mock import MagicMock, patch

import pytest
from llama_index.core.vector_stores.types import (
    BasePydanticVectorStore,
    VectorStoreQuery,
)
from llama_index.vector_stores.supabase import SupabaseVectorStore


def test_class():
    names_of_base_classes = [b.__name__ for b in SupabaseVectorStore.__mro__]
    assert BasePydanticVectorStore.__name__ in names_of_base_classes


def test_query_returns_cosine_similarity():
    with patch("llama_index.vector_stores.supabase.base.vecs.create_client"):
        store = SupabaseVectorStore(
            postgres_connection_string="postgresql://user:pass@localhost:5432/db",
            collection_name="test",
        )

    # vecs returns cosine distance by default
    store._collection = MagicMock()
    store._collection.query.return_value = [
        ("same", 0.0, {"text": "a"}),
        ("far", 1.0, {"text": "b"}),
    ]

    result = store.query(VectorStoreQuery(query_embedding=[0.1, 0.2]))

    assert result.ids == ["same", "far"]
    assert result.similarities == pytest.approx([1.0, 0.0])
