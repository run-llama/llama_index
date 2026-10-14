"""Test keyword table index."""

from pathlib import Path
from typing import Any, List
from unittest.mock import patch

import pytest
from llama_index.core import StorageContext, load_index_from_storage
from llama_index.core.indices.keyword_table.simple_base import (
    SimpleKeywordTableIndex,
)
from llama_index.core.llms import MockLLM
from llama_index.core.schema import Document, TextNode
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


@pytest.mark.parametrize("delete_last_node", [False, True])
@pytest.mark.parametrize("persisted", [False, True])
def test_empty_keyword_table_node_ids(
    delete_last_node: bool, persisted: bool, tmp_path: Path
) -> None:
    nodes = [TextNode(text="apple banana", id_="node-1")] if delete_last_node else []
    index = SimpleKeywordTableIndex(nodes, llm=MockLLM())
    if delete_last_node:
        index.delete_nodes(["node-1"], delete_from_docstore=True)

    if persisted:
        index.storage_context.persist(persist_dir=str(tmp_path))
        storage_context = StorageContext.from_defaults(persist_dir=str(tmp_path))
        index_struct = load_index_from_storage(
            storage_context, llm=MockLLM()
        ).index_struct
    else:
        index_struct = index.index_struct

    assert index_struct.table == {}
    assert index_struct.node_ids == set()


def test_keyword_table_node_ids_are_distinct_from_keyword_sets() -> None:
    index = SimpleKeywordTableIndex(
        [
            TextNode(text="apple banana", id_="node-1"),
            TextNode(text="apple cherry", id_="node-2"),
        ],
        llm=MockLLM(),
    )

    node_ids = index.index_struct.node_ids
    assert node_ids == {"node-1", "node-2"}
    node_ids.clear()
    assert index.index_struct.node_ids == {"node-1", "node-2"}
