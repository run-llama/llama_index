import dataclasses

import pytest

from llama_index.core.vector_stores.types import (
    VectorStoreQuery,
    VectorStoreQueryMode,
)


def test_vector_store_query_rejects_unknown_kwargs() -> None:
    """
    VectorStoreQuery as a strict dataclass should still reject unknown kwargs.
    This validates that we did NOT weaken the dataclass — sanitization
    must happen upstream in the retriever, not here.
    """
    with pytest.raises(TypeError):
        VectorStoreQuery(response_mode="compact", unknown_param=True)


def test_vector_store_query_accepts_valid_kwargs() -> None:
    """All documented fields should still be accepted normally."""
    query = VectorStoreQuery(
        query_str="hello world",
        similarity_top_k=5,
        mode=VectorStoreQueryMode.DEFAULT,
    )
    assert query.query_str == "hello world"
    assert query.similarity_top_k == 5


def test_vsq_field_filter_pattern() -> None:
    """
    Validate that the dataclasses.fields filter used in sub_retrievers
    correctly strips unknown kwargs before VectorStoreQuery construction.
    """
    raw_kwargs = {
        "query_str": "test query",
        "similarity_top_k": 3,
        "response_mode": "compact",  # unknown — should be dropped
        "verbose": True,  # unknown — should be dropped
        "node_postprocessors": [],  # unknown — should be dropped
        "use_async": False,  # unknown — should be dropped
    }
    vsq_fields = {f.name for f in dataclasses.fields(VectorStoreQuery)}
    filtered = {k: v for k, v in raw_kwargs.items() if k in vsq_fields}

    query = VectorStoreQuery(**filtered)  # must not raise
    assert query.query_str == "test query"
    assert query.similarity_top_k == 3
