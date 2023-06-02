"""Test keyword table index."""

from typing import Any, List
from unittest.mock import patch

import pytest

from llama_index.indices.keyword_table.simple_base import GPTSimpleKeywordTableIndex
from llama_index.indices.service_context import ServiceContext
from llama_index.readers.schema.base import Document
from tests.mock_utils.mock_utils import mock_extract_keywords


@pytest.fixture
def documents() -> List[Document]:
    """Get documents."""
    # NOTE: one document for now
    doc_text = (
        "Hello world.\n"
        "This is a test.\n"
        "This is another test.\n"
        "This is a test v2."
    )
    return [Document(doc_text)]


@patch(
    "llama_index.indices.keyword_table.simple_base.simple_extract_keywords",
    mock_extract_keywords,
)
def test_build_table(
    documents: List[Document], mock_service_context: ServiceContext
) -> None:
    """Test build table."""
    # test simple keyword table
    # NOTE: here the keyword extraction isn't mocked because we're using
    # the regex-based keyword extractor, not GPT
    table = GPTSimpleKeywordTableIndex.from_documents(
        documents, service_context=mock_service_context
    )
    nodes = table.docstore.get_nodes(list(table.index_struct.node_ids))
    table_chunks = {n.get_text() for n in nodes}
    assert len(table_chunks) == 4
    assert "Hello world." in table_chunks
    assert "This is a test." in table_chunks
    assert "This is another test." in table_chunks
    assert "This is a test v2." in table_chunks

    # test that expected keys are present in table
    # NOTE: in mock keyword extractor, stopwords are not filtered
    assert table.index_struct.table.keys() == {
        "this",
        "hello",
        "world",
        "test",
        "another",
        "v2",
        "is",
        "a",
        "v2",
    }


@patch(
    "llama_index.indices.keyword_table.simple_base.simple_extract_keywords",
    mock_extract_keywords,
)
def test_build_table_async(
    allow_networking: Any,
    documents: List[Document],
    mock_service_context: ServiceContext,
) -> None:
    """Test build table."""
    # test simple keyword table
    # NOTE: here the keyword extraction isn't mocked because we're using
    # the regex-based keyword extractor, not GPT
    table = GPTSimpleKeywordTableIndex.from_documents(
        documents, use_async=True, service_context=mock_service_context
    )
    nodes = table.docstore.get_nodes(list(table.index_struct.node_ids))
    table_chunks = {n.get_text() for n in nodes}
    assert len(table_chunks) == 4
    assert "Hello world." in table_chunks
    assert "This is a test." in table_chunks
    assert "This is another test." in table_chunks
    assert "This is a test v2." in table_chunks

    # test that expected keys are present in table
    # NOTE: in mock keyword extractor, stopwords are not filtered
    assert table.index_struct.table.keys() == {
        "this",
        "hello",
        "world",
        "test",
        "another",
        "v2",
        "is",
        "a",
        "v2",
    }


@patch(
    "llama_index.indices.keyword_table.simple_base.simple_extract_keywords",
    mock_extract_keywords,
)
def test_insert(
    documents: List[Document],
    mock_service_context: ServiceContext,
) -> None:
    """Test insert."""
    table = GPTSimpleKeywordTableIndex([], service_context=mock_service_context)
    assert len(table.index_struct.table.keys()) == 0
    table.insert(documents[0])
    nodes = table.docstore.get_nodes(list(table.index_struct.node_ids))
    table_chunks = {n.get_text() for n in nodes}
    assert "Hello world." in table_chunks
    assert "This is a test." in table_chunks
    assert "This is another test." in table_chunks
    assert "This is a test v2." in table_chunks
    # test that expected keys are present in table
    # NOTE: in mock keyword extractor, stopwords are not filtered
    assert table.index_struct.table.keys() == {
        "this",
        "hello",
        "world",
        "test",
        "another",
        "v2",
        "is",
        "a",
        "v2",
    }

    # test insert with doc_id
    document1 = Document("This is", doc_id="test_id1")
    document2 = Document("test v3", doc_id="test_id2")
    table = GPTSimpleKeywordTableIndex([])
    table.insert(document1)
    table.insert(document2)
    chunk_index1_1 = list(table.index_struct.table["this"])[0]
    chunk_index1_2 = list(table.index_struct.table["is"])[0]
    chunk_index2_1 = list(table.index_struct.table["test"])[0]
    chunk_index2_2 = list(table.index_struct.table["v3"])[0]
    nodes = table.docstore.get_nodes(
        [chunk_index1_1, chunk_index1_2, chunk_index2_1, chunk_index2_2]
    )
    assert nodes[0].ref_doc_id == "test_id1"
    assert nodes[1].ref_doc_id == "test_id1"
    assert nodes[2].ref_doc_id == "test_id2"
    assert nodes[3].ref_doc_id == "test_id2"


@patch(
    "llama_index.indices.keyword_table.simple_base.simple_extract_keywords",
    mock_extract_keywords,
)
def test_delete(
    mock_service_context: ServiceContext,
) -> None:
    """Test insert."""
    new_documents = [
        Document("Hello world.\nThis is a test.", doc_id="test_id_1"),
        Document("This is another test.", doc_id="test_id_2"),
        Document("This is a test v2.", doc_id="test_id_3"),
    ]

    # test delete
    table = GPTSimpleKeywordTableIndex.from_documents(
        new_documents, service_context=mock_service_context
    )
    # test delete
    table.delete_ref_doc("test_id_1")
    assert len(table.index_struct.table.keys()) == 6
    assert len(table.index_struct.table["this"]) == 2

    # test node contents after delete
    nodes = table.docstore.get_nodes(list(table.index_struct.node_ids))
    node_texts = {n.get_text() for n in nodes}
    assert node_texts == {"This is another test.", "This is a test v2."}

    table = GPTSimpleKeywordTableIndex.from_documents(
        new_documents, service_context=mock_service_context
    )

    # test ref doc info
    all_ref_doc_info = table.ref_doc_info
    for doc_id in all_ref_doc_info.keys():
        assert doc_id in ("test_id_1", "test_id_2", "test_id_3")

    # test delete
    table.delete_ref_doc("test_id_2")
    assert len(table.index_struct.table.keys()) == 7
    assert len(table.index_struct.table["this"]) == 2

    # test node contents after delete
    nodes = table.docstore.get_nodes(list(table.index_struct.node_ids))
    node_texts = {n.get_text() for n in nodes}
    assert node_texts == {"Hello world.", "This is a test.", "This is a test v2."}
