from typing import List, Optional, Tuple

import pytest

from llama_index.core import MockEmbedding
from llama_index.core.graph_stores.simple_labelled import SimplePropertyGraphStore
from llama_index.core.graph_stores.types import EntityNode, LabelledNode, Relation
from llama_index.core.indices.property_graph import VectorContextRetriever
from llama_index.core.schema import QueryBundle, TextNode
from llama_index.core.vector_stores.simple import SimpleVectorStore
from llama_index.core.vector_stores.types import VectorStoreQuery


@pytest.fixture(params=["vector_store", "graph_store"])
def graph_stores(
    request: pytest.FixtureRequest, monkeypatch: pytest.MonkeyPatch
) -> Tuple[SimplePropertyGraphStore, SimpleVectorStore]:
    graph_store = SimplePropertyGraphStore()
    graph_store.upsert_nodes(
        [EntityNode(name=name) for name in ["poor", "good", "left", "right"]]
    )
    graph_store.upsert_relations(
        [
            Relation(label="poor_match", source_id="poor", target_id="right"),
            Relation(label="good_match", source_id="left", target_id="good"),
            Relation(label="both_matched", source_id="poor", target_id="good"),
        ]
    )
    vector_store = SimpleVectorStore()
    vector_store.add(
        [
            TextNode(id_="poor", text="poor", embedding=[-1.0, 0.0]),
            TextNode(id_="good", text="good", embedding=[-3.0, 4.0]),
        ]
    )

    if request.param == "graph_store":
        # Exercise native graph vector queries with the same real cosine scores.
        def vector_query(
            query: VectorStoreQuery,
        ) -> Tuple[List[LabelledNode], List[float]]:
            result = vector_store.query(query)
            assert result.ids is not None and result.similarities is not None
            return [
                graph_store.graph.nodes[id_] for id_ in result.ids
            ], result.similarities

        monkeypatch.setattr(graph_store, "supports_vector_queries", True)
        monkeypatch.setattr(graph_store, "vector_query", vector_query)

    return graph_store, vector_store


@pytest.mark.parametrize("use_async", [False, True])
@pytest.mark.parametrize("threshold", [None, -0.8])
@pytest.mark.asyncio
async def test_negative_endpoint_scores_are_preserved(
    graph_stores: Tuple[SimplePropertyGraphStore, SimpleVectorStore],
    use_async: bool,
    threshold: Optional[float],
) -> None:
    graph_retriever = VectorContextRetriever(
        graph_store=graph_stores[0],
        vector_store=graph_stores[1],
        embed_model=MockEmbedding(embed_dim=2),
        include_text=False,
        similarity_top_k=2,
        similarity_score=threshold,
    )
    query = QueryBundle("query", embedding=[1.0, 0.0])

    if use_async:
        results = await graph_retriever.aretrieve(query)
    else:
        results = graph_retriever.retrieve(query)

    expected = {
        "left -> good_match -> good": -0.6,
        "poor -> both_matched -> good": -0.6,
    }
    if threshold is None:
        expected["poor -> poor_match -> right"] = -1.0

    assert {node.text: node.score for node in results} == pytest.approx(expected)
    assert [node.score for node in results] == sorted(expected.values(), reverse=True)


@pytest.mark.parametrize("use_async", [False, True])
@pytest.mark.asyncio
async def test_unretrieved_endpoints_keep_default_score(
    graph_stores: Tuple[SimplePropertyGraphStore, SimpleVectorStore], use_async: bool
) -> None:
    graph_stores[0].upsert_relations(
        [Relation(label="further", source_id="right", target_id="left")]
    )
    retriever = VectorContextRetriever(
        graph_store=graph_stores[0],
        vector_store=graph_stores[1],
        embed_model=MockEmbedding(embed_dim=2),
        include_text=False,
        similarity_top_k=2,
        path_depth=2,
    )
    query = QueryBundle("query", embedding=[1.0, 0.0])

    results = (
        await retriever.aretrieve(query) if use_async else retriever.retrieve(query)
    )

    further_scores = [
        node.score for node in results if node.text == "right -> further -> left"
    ]
    assert further_scores == [0.0]
