from unittest.mock import MagicMock

import pytest

from llama_index.core import MockEmbedding
from llama_index.core.graph_stores.types import (
    EntityNode,
    PropertyGraphStore,
    Relation,
)
from llama_index.core.indices.property_graph.sub_retrievers.vector import (
    VectorContextRetriever,
)
from llama_index.core.schema import QueryBundle


def _mock_vector_graph_store() -> MagicMock:
    low_left = EntityNode(name="low-left")
    low_right = EntityNode(name="low-right")
    high_left = EntityNode(name="high-left")
    high_right = EntityNode(name="high-right")
    low_relation = Relation(
        label="low-relation", source_id=low_left.id, target_id=low_right.id
    )
    high_relation = Relation(
        label="high-relation", source_id=high_left.id, target_id=high_right.id
    )
    nodes = [low_left, low_right, high_left, high_right]
    scores = [-0.4, -0.2, 0.1, 0.3]
    triplets = [
        (low_left, low_relation, low_right),
        (high_left, high_relation, high_right),
    ]

    graph_store = MagicMock(spec=PropertyGraphStore)
    graph_store.supports_vector_queries = True
    graph_store.vector_query.return_value = (nodes, scores)
    graph_store.get_rel_map.return_value = triplets
    graph_store.avector_query.return_value = (nodes, scores)
    graph_store.aget_rel_map.return_value = triplets
    return graph_store


def _zero_threshold_retriever(
    graph_store: PropertyGraphStore,
) -> VectorContextRetriever:
    return VectorContextRetriever(
        graph_store=graph_store,
        embed_model=MockEmbedding(embed_dim=1),
        include_text=False,
        similarity_score=0.0,
    )


def test_vector_context_retriever_applies_zero_threshold() -> None:
    retriever = _zero_threshold_retriever(_mock_vector_graph_store())

    results = retriever.retrieve_from_graph(QueryBundle("query", embedding=[1.0]))

    assert [result.score for result in results] == [0.3]


@pytest.mark.asyncio
async def test_vector_context_retriever_applies_zero_threshold_async() -> None:
    retriever = _zero_threshold_retriever(_mock_vector_graph_store())

    results = await retriever.aretrieve_from_graph(
        QueryBundle("query", embedding=[1.0])
    )

    assert [result.score for result in results] == [0.3]
