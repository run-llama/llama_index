"""Test keyword table index."""

from typing import Any, List
from unittest.mock import patch

import pytest
from llama_index.core.indices.keyword_table.simple_base import (
    SimpleKeywordTableIndex,
)
from llama_index.core.schema import Document
from tests.mock_utils.mock_utils import mock_extract_keywords


@pytest.fixture()
def documents() -> List[Document]:
    """Get documents."""
    # NOTE: one document for now
    doc_text = (
        "Hello world.\nThis is a test.\nThis is another test.\nThis is a test v2."
    )
    return [Document(text=doc_text)]


@patch(
    "llama_index.core.indices.keyword_table.simple_base.simple_extract_keywords",
    mock_extract_keywords,
)
def test_build_table(documents: List[Document], patch_token_text_splitter) -> None:
    """Test build table."""
    # test simple keyword table
    # NOTE: here the keyword extraction isn't mocked because we're using
    # the regex-based keyword extractor, not GPT
    table = SimpleKeywordTableIndex.from_documents(documents)
    nodes = table.docstore.get_nodes(list(table.index_struct.node_ids))
    table_chunks = {n.get_content() for n in nodes}
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
    "llama_index.core.indices.keyword_table.simple_base.simple_extract_keywords",
    mock_extract_keywords,
)
def test_build_table_async(
    allow_networking: Any, documents: List[Document], patch_token_text_splitter
) -> None:
    """Test build table."""
    # test simple keyword table
    # NOTE: here the keyword extraction isn't mocked because we're using
    # the regex-based keyword extractor, not GPT
    table = SimpleKeywordTableIndex.from_documents(documents, use_async=True)
    nodes = table.docstore.get_nodes(list(table.index_struct.node_ids))
    table_chunks = {n.get_content() for n in nodes}
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
    "llama_index.core.indices.keyword_table.simple_base.simple_extract_keywords",
    mock_extract_keywords,
)
def test_insert(documents: List[Document], patch_token_text_splitter) -> None:
    """Test insert."""
    table = SimpleKeywordTableIndex([])
    assert len(table.index_struct.table.keys()) == 0
    table.insert(documents[0])
    nodes = table.docstore.get_nodes(list(table.index_struct.node_ids))
    table_chunks = {n.get_content() for n in nodes}
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
    document1 = Document(text="This is", id_="test_id1")
    document2 = Document(text="test v3", id_="test_id2")
    table = SimpleKeywordTableIndex([])
    table.insert(document1)
    table.insert(document2)
    chunk_index1_1 = next(iter(table.index_struct.table["this"]))
    chunk_index1_2 = next(iter(table.index_struct.table["is"]))
    chunk_index2_1 = next(iter(table.index_struct.table["test"]))
    chunk_index2_2 = next(iter(table.index_struct.table["v3"]))
    nodes = table.docstore.get_nodes(
        [
            chunk_index1_1,
            chunk_index1_2,
            chunk_index2_1,
            chunk_index2_2,
        ]
    )
    assert nodes[0].ref_doc_id == "test_id1"
    assert nodes[1].ref_doc_id == "test_id1"
    assert nodes[2].ref_doc_id == "test_id2"
    assert nodes[3].ref_doc_id == "test_id2"


@patch(
    "llama_index.core.indices.keyword_table.simple_base.simple_extract_keywords",
    mock_extract_keywords,
)
def test_delete(patch_token_text_splitter) -> None:
    """Test insert."""
    new_documents = [
        Document(text="Hello world.\nThis is a test.", id_="test_id_1"),
        Document(text="This is another test.", id_="test_id_2"),
        Document(text="This is a test v2.", id_="test_id_3"),
    ]

    # test delete
    table = SimpleKeywordTableIndex.from_documents(new_documents)
    # test delete
    table.delete_ref_doc("test_id_1")
    assert len(table.index_struct.table.keys()) == 6
    assert len(table.index_struct.table["this"]) == 2

    # test node contents after delete
    nodes = table.docstore.get_nodes(list(table.index_struct.node_ids))
    node_texts = {n.get_content() for n in nodes}
    assert node_texts == {"This is another test.", "This is a test v2."}

    table = SimpleKeywordTableIndex.from_documents(new_documents)

    # test ref doc info
    all_ref_doc_info = table.ref_doc_info
    for doc_id in all_ref_doc_info:
        assert doc_id in ("test_id_1", "test_id_2", "test_id_3")

    # test delete
    table.delete_ref_doc("test_id_2")
    assert len(table.index_struct.table.keys()) == 7
    assert len(table.index_struct.table["this"]) == 2

    # test node contents after delete
    nodes = table.docstore.get_nodes(list(table.index_struct.node_ids))
    node_texts = {n.get_content() for n in nodes}
    assert node_texts == {"Hello world.", "This is a test.", "This is a test v2."}
