"""Test keyword table index."""

from typing import Any, List
from unittest.mock import patch

import pytest

from gpt_index.indices.keyword_table.simple_base import GPTSimpleKeywordTableIndex
from gpt_index.readers.schema.base import Document
from tests.mock_utils.mock_decorator import patch_common
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


@patch_common
@patch(
    "gpt_index.indices.keyword_table.simple_base.simple_extract_keywords",
    mock_extract_keywords,
)
def test_build_table(
    _mock_init: Any,
    _mock_predict: Any,
    _mock_total_tokens_used: Any,
    _mock_split_text: Any,
    documents: List[Document],
) -> None:
    """Test build table."""
    # test simple keyword table
    # NOTE: here the keyword extraction isn't mocked because we're using
    # the regex-based keyword extractor, not GPT
    table = GPTSimpleKeywordTableIndex(documents)
    table_chunks = {n.text for n in table.index_struct.text_chunks.values()}
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


@patch_common
@patch(
    "gpt_index.indices.keyword_table.simple_base.simple_extract_keywords",
    mock_extract_keywords,
)
def test_insert(
    _mock_init: Any,
    _mock_predict: Any,
    _mock_total_tokens_used: Any,
    _mock_split_text: Any,
    documents: List[Document],
) -> None:
    """Test insert."""
    table = GPTSimpleKeywordTableIndex([])
    assert len(table.index_struct.table.keys()) == 0
    table.insert(documents[0])
    table_chunks = {n.text for n in table.index_struct.text_chunks.values()}
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
    assert table.index_struct.text_chunks[chunk_index1_1].ref_doc_id == "test_id1"
    assert table.index_struct.text_chunks[chunk_index1_2].ref_doc_id == "test_id1"
    assert table.index_struct.text_chunks[chunk_index2_1].ref_doc_id == "test_id2"
    assert table.index_struct.text_chunks[chunk_index2_2].ref_doc_id == "test_id2"
