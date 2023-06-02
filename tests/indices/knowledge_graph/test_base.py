"""Test knowledge graph index."""

from typing import Any, Dict, List, Tuple
from unittest.mock import patch

import pytest
from llama_index.data_structs.node import Node
from llama_index.embeddings.base import BaseEmbedding
from llama_index.indices.knowledge_graph.base import GPTKnowledgeGraphIndex
from llama_index.indices.service_context import ServiceContext
from llama_index.readers.schema.base import Document
from tests.mock_utils.mock_prompts import (
    MOCK_KG_TRIPLET_EXTRACT_PROMPT,
    MOCK_QUERY_KEYWORD_EXTRACT_PROMPT,
)


class MockEmbedding(BaseEmbedding):
    def _get_text_embedding(self, text: str) -> List[float]:
        """Mock get text embedding."""
        # assume dimensions are 4
        if text == "('foo', 'is', 'bar')":
            return [1, 0, 0, 0]
        elif text == "('hello', 'is not', 'world')":
            return [0, 1, 0, 0]
        elif text == "('Jane', 'is mother of', 'Bob')":
            return [0, 0, 1, 0]
        elif text == "foo":
            return [0, 0, 0, 1]
        else:
            raise ValueError("Invalid text for `mock_get_text_embedding`.")

    def _get_query_embedding(self, query: str) -> List[float]:
        """Mock get query embedding."""
        del query
        return [0, 0, 1, 0, 0]


@pytest.fixture
def struct_kwargs() -> Tuple[Dict, Dict]:
    """Index kwargs."""
    index_kwargs = {
        "kg_triple_extract_template": MOCK_KG_TRIPLET_EXTRACT_PROMPT,
    }
    query_kwargs = {
        "query_keyword_extract_template": MOCK_QUERY_KEYWORD_EXTRACT_PROMPT,
    }
    return index_kwargs, query_kwargs


def mock_extract_triplets(text: str) -> List[Tuple[str, str, str]]:
    """Mock extract triplets."""
    lines = text.split("\n")
    triplets: List[Tuple[str, str, str]] = []
    for line in lines:
        tokens = line[1:-1].split(",")
        tokens = [t.strip() for t in tokens]

        subj, pred, obj = tokens
        triplets.append((subj, pred, obj))
    return triplets


@patch.object(
    GPTKnowledgeGraphIndex, "_extract_triplets", side_effect=mock_extract_triplets
)
def test_build_kg_manual(
    _patch_extract_triplets: Any,
    mock_service_context: ServiceContext,
) -> None:
    """Test build knowledge graph."""
    index = GPTKnowledgeGraphIndex([], service_context=mock_service_context)
    tuples = [
        ("foo", "is", "bar"),
        ("hello", "is not", "world"),
        ("Jane", "is mother of", "Bob"),
    ]
    nodes = [Node(str(tup)) for tup in tuples]
    for tup, node in zip(tuples, nodes):
        # add node
        index.add_node([tup[0], tup[2]], node)
        # add triplet
        index.upsert_triplet(tup)

    # NOTE: in these unit tests, document text == triplets
    nodes = index.docstore.get_nodes(list(index.index_struct.node_ids))
    table_chunks = {n.get_text() for n in nodes}
    assert len(table_chunks) == 3
    assert "('foo', 'is', 'bar')" in table_chunks
    assert "('hello', 'is not', 'world')" in table_chunks
    assert "('Jane', 'is mother of', 'Bob')" in table_chunks

    # test that expected keys are present in table
    # NOTE: in mock keyword extractor, stopwords are not filtered
    assert index.index_struct.table.keys() == {
        "foo",
        "bar",
        "hello",
        "world",
        "Jane",
        "Bob",
    }

    # test upsert_triplet_and_node
    index = GPTKnowledgeGraphIndex([], service_context=mock_service_context)
    tuples = [
        ("foo", "is", "bar"),
        ("hello", "is not", "world"),
        ("Jane", "is mother of", "Bob"),
    ]
    nodes = [Node(str(tup)) for tup in tuples]
    for tup, node in zip(tuples, nodes):
        index.upsert_triplet_and_node(tup, node)

    # NOTE: in these unit tests, document text == triplets
    nodes = index.docstore.get_nodes(list(index.index_struct.node_ids))
    table_chunks = {n.get_text() for n in nodes}
    assert len(table_chunks) == 3
    assert "('foo', 'is', 'bar')" in table_chunks
    assert "('hello', 'is not', 'world')" in table_chunks
    assert "('Jane', 'is mother of', 'Bob')" in table_chunks

    # test that expected keys are present in table
    # NOTE: in mock keyword extractor, stopwords are not filtered
    assert index.index_struct.table.keys() == {
        "foo",
        "bar",
        "hello",
        "world",
        "Jane",
        "Bob",
    }

    # try inserting same node twice
    index = GPTKnowledgeGraphIndex([], service_context=mock_service_context)
    node = Node(str(("foo", "is", "bar")), doc_id="test_node")
    index.upsert_triplet_and_node(tup, node)
    index.upsert_triplet_and_node(tup, node)


@patch.object(
    GPTKnowledgeGraphIndex, "_extract_triplets", side_effect=mock_extract_triplets
)
def test_build_kg_similarity(
    _patch_extract_triplets: Any,
    documents: List[Document],
    mock_service_context: ServiceContext,
) -> None:
    """Test build knowledge graph."""
    mock_service_context.embed_model = MockEmbedding()

    index = GPTKnowledgeGraphIndex.from_documents(
        documents, include_embeddings=True, service_context=mock_service_context
    )
    # get embedding dict from KG index struct
    rel_text_embeddings = index.index_struct.embedding_dict

    # check that all rel_texts were embedded
    assert len(rel_text_embeddings) == 3
    for rel_text, embedding in rel_text_embeddings.items():
        assert embedding == MockEmbedding().get_text_embedding(rel_text)


@patch.object(
    GPTKnowledgeGraphIndex, "_extract_triplets", side_effect=mock_extract_triplets
)
def test_build_kg(
    _patch_extract_triplets: Any,
    documents: List[Document],
    mock_service_context: ServiceContext,
) -> None:
    """Test build knowledge graph."""
    index = GPTKnowledgeGraphIndex.from_documents(
        documents, service_context=mock_service_context
    )
    # NOTE: in these unit tests, document text == triplets
    nodes = index.docstore.get_nodes(list(index.index_struct.node_ids))
    table_chunks = {n.get_text() for n in nodes}
    assert len(table_chunks) == 3
    assert "(foo, is, bar)" in table_chunks
    assert "(hello, is not, world)" in table_chunks
    assert "(Jane, is mother of, Bob)" in table_chunks

    # test that expected keys are present in table
    # NOTE: in mock keyword extractor, stopwords are not filtered
    assert index.index_struct.table.keys() == {
        "foo",
        "bar",
        "hello",
        "world",
        "Jane",
        "Bob",
    }

    # test ref doc info for three nodes, single doc
    all_ref_doc_info = index.ref_doc_info
    assert len(all_ref_doc_info) == 1
    for ref_doc_info in all_ref_doc_info.values():
        assert len(ref_doc_info.doc_ids) == 3
