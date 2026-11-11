from unittest.mock import MagicMock
from pathlib import Path
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
from llama_index.core.indices.property_graph.transformations import (
    ImplicitPathExtractor,
)
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


@pytest.mark.parametrize("use_async", [False, True])
@pytest.mark.parametrize("reload_store", [False, True])
def test_insert_nodes_with_swapped_content(
    use_async: bool, reload_store: bool, tmp_path: Path
) -> None:
    index = PropertyGraphIndex(
        nodes=[TextNode(id_="a", text="alpha"), TextNode(id_="b", text="beta")],
        kg_extractors=[ImplicitPathExtractor()],
        embed_kg_nodes=False,
        use_async=use_async,
    )

    if reload_store:
        persist_path = str(tmp_path / "graph.json")
        index.property_graph_store.persist(persist_path)
        index = PropertyGraphIndex.from_existing(
            SimplePropertyGraphStore.from_persist_path(persist_path),
            kg_extractors=[ImplicitPathExtractor()],
            embed_kg_nodes=False,
            use_async=use_async,
        )

    index.insert_nodes(
        [TextNode(id_="a", text="beta"), TextNode(id_="b", text="alpha")]
    )

    stored = index.property_graph_store.get_llama_nodes(["a", "b"])
    assert {node.node_id: node.text for node in stored} == {"a": "beta", "b": "alpha"}


@pytest.mark.parametrize("use_async", [False, True])
def test_insert_nodes_with_same_content_and_distinct_ids(use_async: bool) -> None:
    index = PropertyGraphIndex(
        nodes=[TextNode(id_="a", text="shared", embedding=[1.0, 0.0])],
        kg_extractors=[ImplicitPathExtractor()],
        embed_kg_nodes=False,
        use_async=use_async,
    )

    index.insert_nodes(
        [
            TextNode(id_="a", text="shared", embedding=[0.5, 0.5]),
            TextNode(id_="b", text="shared", embedding=[0.0, 1.0]),
        ]
    )

    stored = index.property_graph_store.get_llama_nodes(["a", "b"])
    assert {node.node_id: node.text for node in stored} == {
        "a": "shared",
        "b": "shared",
    }
    # The same ID and content is still skipped; a distinct ID must be inserted.
    graph_nodes = index.property_graph_store.get(ids=["a", "b"])
    assert {node.id: node.embedding for node in graph_nodes} == {
        "a": [1.0, 0.0],
        "b": [0.0, 1.0],
    }
