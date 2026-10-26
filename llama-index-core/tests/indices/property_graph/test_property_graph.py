from unittest.mock import MagicMock
from typing import Any, List

from llama_index.core import PropertyGraphIndex, Document, MockEmbedding
from llama_index.core.graph_stores.simple_labelled import SimplePropertyGraphStore
from llama_index.core.graph_stores.types import (
    EntityNode,
    Relation,
    KG_NODES_KEY,
    KG_RELATIONS_KEY,
)
from llama_index.core.llms.mock import MockLLM
from llama_index.core.schema import BaseNode, TextNode, TransformComponent
from llama_index.core.vector_stores.simple import SimpleVectorStore


class MockKGExtractor(TransformComponent):
    """A mock knowledge graph extractor that extracts a simple relation from a text."""

    def __call__(self, nodes: List[BaseNode], **kwargs: Any) -> List[BaseNode]:
        entity1 = EntityNode(name="Logan", label="PERSON")
        entity2 = EntityNode(name="Canada", label="LOCATION")
        relation = Relation(label="BORN_IN", source_id=entity1.id, target_id=entity2.id)

        return [
            TextNode(
                id_="test",
                text="Logan was born in Canada",
                metadata={
                    KG_NODES_KEY: [entity1, entity2],
                    KG_RELATIONS_KEY: [relation],
                },
            ),
        ]


def test_construction() -> None:
    graph_store = SimplePropertyGraphStore()
    vector_store = SimpleVectorStore()
    kg_extractor = MockKGExtractor()

    # test construction
    index = PropertyGraphIndex.from_documents(
        [Document.example()],
        property_graph_store=graph_store,
        vector_store=vector_store,
        llm=MockLLM(),
        embed_model=MockEmbedding(embed_dim=256),
        kg_extractors=[kg_extractor],
    )

    embeddings = vector_store.get("Logan")
    assert len(embeddings) == 256

    embeddings = vector_store.get("Canada")
    assert len(embeddings) == 256

    kg_nodes = graph_store.get(ids=["Logan", "Canada"])
    assert kg_nodes is not None
    assert len(kg_nodes) == 2
    assert kg_nodes[0].embedding is None
    assert kg_nodes[0].embedding is None

    # test inserting a duplicate node (should not insert)
    index._insert_nodes_to_vector_index = MagicMock()
    index.insert_nodes(kg_extractor([]))

    assert index._insert_nodes_to_vector_index.call_count == 0


def test_base_pg_retriever_add_source_text_deduplicates_and_caches() -> None:
    import asyncio
    from llama_index.core.indices.property_graph.sub_retrievers.base import (
        BasePGRetriever,
    )
    from llama_index.core.schema import (
        NodeRelationship,
        NodeWithScore,
        QueryBundle,
        RelatedNodeInfo,
    )

    class DummyPGRetriever(BasePGRetriever):
        def retrieve_from_graph(
            self, query_bundle: QueryBundle
        ) -> List[NodeWithScore]:
            return []

        async def aretrieve_from_graph(
            self, query_bundle: QueryBundle
        ) -> List[NodeWithScore]:
            return []

    graph_store = SimplePropertyGraphStore()
    chunk_1 = TextNode(id_="doc-1", text="Original chunk 1 text.")
    chunk_2 = TextNode(id_="doc-2", text="Original chunk 2 text.")
    graph_store.upsert_llama_nodes([chunk_1, chunk_2])

    retriever = DummyPGRetriever(graph_store=graph_store, include_text=True)

    retrieved = [
        NodeWithScore(
            node=TextNode(
                text=f"Entity{i} -> REL -> Target{i}",
                relationships={
                    NodeRelationship.SOURCE: RelatedNodeInfo(
                        node_id="doc-1" if i < 3 else "doc-2"
                    )
                },
            ),
            score=1.0 - i * 0.1,
        )
        for i in range(5)
    ]

    orig_get = graph_store.get_llama_nodes
    get_spy = MagicMock(wraps=orig_get)
    graph_store.get_llama_nodes = get_spy  # type: ignore[method-assign]

    results = retriever.add_source_text(retrieved)
    assert get_spy.call_count == 1
    assert get_spy.call_args.args[0] == ["doc-1", "doc-2"]
    assert len(results) == 5
    # Nodes sharing the same source chunk reuse the constructed TextNode instance
    assert results[0].node is results[1].node
    assert results[1].node is results[2].node
    assert results[3].node is results[4].node
    assert "Original chunk 1 text." in results[0].node.get_content()
    assert "Entity0 -> REL -> Target0" in results[0].node.get_content()
    assert "Entity2 -> REL -> Target2" in results[0].node.get_content()

    orig_aget = graph_store.aget_llama_nodes
    aget_spy = MagicMock(wraps=orig_aget)
    graph_store.aget_llama_nodes = aget_spy  # type: ignore[method-assign]

    async_results = asyncio.run(retriever.async_add_source_text(retrieved))
    assert aget_spy.call_count == 1
    assert aget_spy.call_args.args[0] == ["doc-1", "doc-2"]
    assert async_results[0].node is async_results[1].node
    assert async_results[3].node is async_results[4].node

