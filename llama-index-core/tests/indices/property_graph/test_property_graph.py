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

class MockMultiDocDuplicateEntityExtractor(TransformComponent):
    """Extract the same entity id from multiple source nodes in one batch."""

    def __call__(self, nodes: List[BaseNode], **kwargs: Any) -> List[BaseNode]:
        out: List[BaseNode] = []
        for i, node in enumerate(nodes):
            out.append(
                TextNode(
                    id_=f"src-{i}",
                    text=node.get_content() if hasattr(node, "get_content") else str(node),
                    metadata={
                        KG_NODES_KEY: [EntityNode(name="Logan", label="PERSON")],
                        KG_RELATIONS_KEY: [],
                    },
                )
            )
        return out


def test_within_batch_kg_entity_dedup() -> None:
    """Same entity extracted from multiple sources inserts once (#22394)."""
    graph_store = SimplePropertyGraphStore()
    vector_store = SimpleVectorStore()
    kg_extractor = MockMultiDocDuplicateEntityExtractor()

    index = PropertyGraphIndex.from_documents(
        [
            Document(text="Logan lives in Toronto."),
            Document(text="Logan works in Canada."),
        ],
        property_graph_store=graph_store,
        vector_store=vector_store,
        llm=MockLLM(),
        embed_model=MockEmbedding(embed_dim=256),
        kg_extractors=[kg_extractor],
    )

    kg_nodes = graph_store.get(ids=["Logan"])
    assert len(kg_nodes) == 1

    # vector store should also hold a single Logan entry (not two embeddings)
    embeddings = vector_store.get("Logan")
    assert len(embeddings) == 256

