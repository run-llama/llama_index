"""Offline tests for queries that carry no embedding."""

from typing import Any, List, Optional
from unittest.mock import MagicMock, patch

import pytest
from llama_index.core.vector_stores.types import VectorStoreQuery, VectorStoreQueryMode
from llama_index.vector_stores.pinecone import PineconeVectorStore

DEFAULT_VECTOR = [0.5, 0.25, 0.125]


def _store(default_empty_query_vector: Optional[List[float]] = None) -> Any:
    index = MagicMock()
    index.describe_index_stats.return_value = {"dimension": 3}
    index.query.return_value = MagicMock(matches=[])
    store = PineconeVectorStore(
        pinecone_index=index, default_empty_query_vector=default_empty_query_vector
    )
    return store, index


def _sent_vector(index: MagicMock) -> List[float]:
    return index.query.call_args.kwargs["vector"]


@pytest.mark.parametrize(
    "mode", [VectorStoreQueryMode.DEFAULT, VectorStoreQueryMode.HYBRID]
)
def test_default_vector_used_when_query_has_no_embedding(
    mode: VectorStoreQueryMode,
) -> None:
    store, index = _store(DEFAULT_VECTOR)
    store.query(VectorStoreQuery(query_embedding=None, mode=mode, similarity_top_k=2))
    assert _sent_vector(index) == DEFAULT_VECTOR
    index.describe_index_stats.assert_not_called()


def test_query_embedding_takes_precedence_over_default() -> None:
    store, index = _store(DEFAULT_VECTOR)
    store.query(VectorStoreQuery(query_embedding=[1.0, 2.0, 3.0], similarity_top_k=2))
    assert _sent_vector(index) == [1.0, 2.0, 3.0]


def test_alpha_scales_the_default_vector() -> None:
    store, index = _store([1.0, 2.0, 4.0])
    store.query(
        VectorStoreQuery(
            query_embedding=None,
            mode=VectorStoreQueryMode.HYBRID,
            alpha=0.5,
            query_str="q",
            similarity_top_k=2,
        )
    )
    assert _sent_vector(index) == [0.5, 1.0, 2.0]


@pytest.mark.parametrize(
    "mode", [VectorStoreQueryMode.DEFAULT, VectorStoreQueryMode.HYBRID]
)
def test_zero_vector_without_a_default(mode: VectorStoreQueryMode) -> None:
    store, index = _store(None)
    store.query(VectorStoreQuery(query_embedding=None, mode=mode, similarity_top_k=2))
    # a None vector would be rejected by Pinecone; fall back to the index dimension
    assert _sent_vector(index) == [0.0, 0.0, 0.0]


def test_from_params_forwards_the_default_vector() -> None:
    index = MagicMock()
    with patch.object(
        PineconeVectorStore, "_initialize_pinecone_client", return_value=index
    ):
        store = PineconeVectorStore.from_params(
            api_key="key",
            index_name="idx",
            default_empty_query_vector=DEFAULT_VECTOR,
        )
    assert store._default_empty_query_vector == DEFAULT_VECTOR


def test_default_vector_is_optional() -> None:
    store, _ = _store()
    assert store._default_empty_query_vector is None
