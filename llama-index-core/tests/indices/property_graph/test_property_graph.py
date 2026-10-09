from unittest.mock import MagicMock
from typing import Any, List

import pytest

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


@pytest.mark.asyncio
async def test_ainsert_nodes_no_nested_event_loop_error() -> None:
    """
    ainsert_nodes must not call asyncio.run() internally.

    If it did, this test would raise 'RuntimeError: This event loop is already
    running' because pytest-asyncio already runs us inside an event loop.
    We build the index with an empty node list so the sync constructor path
    completes without touching asyncio, then exercise the async insertion path.
    """
    graph_store = SimplePropertyGraphStore()
    vector_store = SimpleVectorStore()
    kg_extractor = MockKGExtractor()

    # nodes=[] → _insert_nodes returns immediately, no asyncio.run() in __init__
    index = PropertyGraphIndex(
        nodes=[],
        property_graph_store=graph_store,
        vector_store=vector_store,
        llm=MockLLM(),
        embed_model=MockEmbedding(embed_dim=256),
        kg_extractors=[kg_extractor],
        use_async=True,
        show_progress=False,
    )

    source_node = TextNode(id_="test1", text="Logan was born in Canada")
    # Before the fix this raised RuntimeError: This event loop is already running
    await index.ainsert_nodes([source_node])


@pytest.mark.asyncio
async def test_ainsert_nodes_inserts_graph_data() -> None:
    """ainsert_nodes should populate the graph store and vector store."""
    graph_store = SimplePropertyGraphStore()
    vector_store = SimpleVectorStore()
    kg_extractor = MockKGExtractor()

    index = PropertyGraphIndex(
        nodes=[],
        property_graph_store=graph_store,
        vector_store=vector_store,
        llm=MockLLM(),
        embed_model=MockEmbedding(embed_dim=256),
        kg_extractors=[kg_extractor],
        use_async=True,
        show_progress=False,
    )

    source_node = TextNode(id_="src1", text="Logan was born in Canada")
    await index.ainsert_nodes([source_node])

    kg_nodes = graph_store.get(ids=["Logan", "Canada"])
    assert len(kg_nodes) == 2

    logan_embedding = vector_store.get("Logan")
    assert len(logan_embedding) == 256


@pytest.mark.asyncio
async def test_ainsert_nodes_skips_duplicate_nodes() -> None:
    """ainsert_nodes should not re-vector-insert nodes that already exist."""
    graph_store = SimplePropertyGraphStore()
    vector_store = SimpleVectorStore()
    kg_extractor = MockKGExtractor()

    index = PropertyGraphIndex(
        nodes=[],
        property_graph_store=graph_store,
        vector_store=vector_store,
        llm=MockLLM(),
        embed_model=MockEmbedding(embed_dim=256),
        kg_extractors=[kg_extractor],
        use_async=True,
        show_progress=False,
    )

    source_node = TextNode(id_="src1", text="Logan was born in Canada")
    await index.ainsert_nodes([source_node])

    # Replace with mock AFTER first insert so we can assert it's NOT called again
    index._ainsert_nodes_to_vector_index = MagicMock()
    # Same node → already in docstore, so _ainsert_nodes_to_vector_index skipped
    await index.ainsert_nodes([source_node])

    assert index._ainsert_nodes_to_vector_index.call_count == 0
