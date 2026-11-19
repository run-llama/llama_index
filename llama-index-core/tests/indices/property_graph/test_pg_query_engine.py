import dataclasses

import pytest

from llama_index.core.indices.property_graph.retriever import PGRetriever
from llama_index.core.schema import NodeWithScore, TextNode
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


def test_pg_retriever_deduplicate_retains_highest_score_and_empty_text_nodes() -> None:
    """PGRetriever._deduplicate should keep the highest score for duplicate texts and preserve distinct empty-text nodes."""
    retriever = PGRetriever(sub_retrievers=[], use_async=False)

    nodes = [
        NodeWithScore(node=TextNode(id_="n1", text="A -> B -> C"), score=None),
        NodeWithScore(node=TextNode(id_="n2", text="X -> Y -> Z"), score=0.4),
        NodeWithScore(node=TextNode(id_="n3", text="A -> B -> C"), score=0.75),
        NodeWithScore(node=TextNode(id_="n4", text="X -> Y -> Z"), score=0.92),
        NodeWithScore(node=TextNode(id_="n5", text="X -> Y -> Z"), score=0.15),
        NodeWithScore(node=TextNode(id_="empty_1", text=""), score=0.5),
        NodeWithScore(node=TextNode(id_="empty_2", text=""), score=0.8),
    ]

    deduped = retriever._deduplicate(nodes)
    assert len(deduped) == 4
    assert [n.node.node_id for n in deduped] == ["n3", "n4", "empty_1", "empty_2"]
    assert [n.score for n in deduped] == [0.75, 0.92, 0.5, 0.8]
